# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_TIMING_FIELDS = (
    "queue_ms",
    "queue_before_remote_kv_ms",
    "queue_after_remote_kv_ms",
    "local_kv_lookup_ms",
    "external_kv_lookup_ms",
    "remote_kv_wait_ms",
    "kv_compute_wall_ms",
    "other_backend_ms",
    "backend_ttft_ms",
)
_REDACTED_PARAM_KEYS = {
    "all_token_ids",
    "hash_ids",
    "input_ids",
    "output_token_ids",
    "prompt_token_ids",
    "token_ids",
}


@dataclass(slots=True)
class _TraceConfig:
    enabled: bool
    path: str | None
    instance_id: str
    configured_role: str


@dataclass(slots=True)
class PDTraceState:
    request_id: str
    instance_id: str
    configured_role: str
    observed_role: str
    pd_coordination_id: str
    transfer_id: str | None
    connector: str | None
    kv_transfer_params: dict[str, Any]
    prompt_tokens: int
    timing_ms: dict[str, float] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    local_cached_tokens: int = 0
    external_cached_tokens: int = 0
    computed_tokens: int = 0
    computed_kv_block_ids: dict[int, set[int]] = field(default_factory=dict)
    remote_requested_kv_block_ids: dict[int, set[int]] = field(default_factory=dict)
    first_lookup_seen: bool = False
    remote_wait_start_ts: float | None = None
    remote_wait_done_ts: float | None = None
    kv_compute_start_ts: float | None = None
    emitted: bool = False


_CONFIG: _TraceConfig | None = None
_TRACE_FILE: TextIO | None = None
_DISABLED_REASON: str | None = None


def enabled() -> bool:
    return _get_config().enabled and _DISABLED_REASON is None


def stage_start(request: Any, connector_name: str | None = None) -> float | None:
    if not enabled():
        return None
    _ensure_state(request, connector_name=connector_name)
    return time.perf_counter()


def stage_end(
    request: Any,
    start_ts: float | None,
    field: str,
    connector_name: str | None = None,
) -> None:
    if start_ts is None or not enabled():
        return
    state = _ensure_state(request, connector_name=connector_name)
    _add_timing(state, field, (time.perf_counter() - start_ts) * 1000.0)


def record_lookup_start(request: Any, connector_name: str | None = None) -> None:
    if not enabled():
        return
    state = _ensure_state(request, connector_name=connector_name)
    if state.first_lookup_seen:
        return
    state.first_lookup_seen = True
    arrival_time = getattr(request, "arrival_time", None)
    if isinstance(arrival_time, (int, float)):
        queue_before_ms = max(0.0, (time.time() - float(arrival_time)) * 1000.0)
        state.timing_ms["queue_before_remote_kv_ms"] = queue_before_ms
        state.timing_ms["queue_ms"] = queue_before_ms


def record_token_sources(
    request: Any,
    *,
    local_cached_tokens: int,
    external_cached_tokens: int,
    load_kv_async: bool,
    connector_name: str | None = None,
) -> None:
    if not enabled():
        return
    state = _ensure_state(request, connector_name=connector_name)
    prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
    state.prompt_tokens = prompt_tokens
    state.local_cached_tokens = int(local_cached_tokens)
    state.external_cached_tokens = int(external_cached_tokens)
    state.computed_tokens = max(
        0, prompt_tokens - state.local_cached_tokens - state.external_cached_tokens
    )
    state.flags["load_kv_async"] = bool(load_kv_async)


def mark_remote_wait_start(
    request: Any, connector_name: str | None = None
) -> None:
    if not enabled():
        return
    state = _ensure_state(request, connector_name=connector_name)
    state.flags["load_kv_async"] = True
    if state.remote_wait_start_ts is None:
        state.remote_wait_start_ts = time.perf_counter()


def mark_remote_wait_done(
    request: Any, connector_name: str | None = None
) -> None:
    if not enabled():
        return
    state = _ensure_state(request, connector_name=connector_name)
    if state.remote_wait_start_ts is None:
        return
    now = time.perf_counter()
    _add_timing(state, "remote_kv_wait_ms", (now - state.remote_wait_start_ts) * 1000.0)
    state.remote_wait_start_ts = None
    state.remote_wait_done_ts = now


def mark_compute_start(
    request: Any,
    *,
    num_new_tokens: int,
    is_chunked_prefill: bool,
    connector_name: str | None = None,
) -> None:
    if num_new_tokens <= 0 or not enabled():
        return
    state = _ensure_state(request, connector_name=connector_name)
    if state.kv_compute_start_ts is not None:
        return
    now = time.perf_counter()
    state.kv_compute_start_ts = now
    if state.remote_wait_done_ts is not None:
        queue_after_ms = max(0.0, (now - state.remote_wait_done_ts) * 1000.0)
        state.timing_ms["queue_after_remote_kv_ms"] = queue_after_ms
        state.timing_ms["queue_ms"] = (
            state.timing_ms.get("queue_before_remote_kv_ms", 0.0)
            + queue_after_ms
        )
    state.flags["chunked_prefill"] = bool(is_chunked_prefill)
    state.flags["preempted"] = int(getattr(request, "num_preemptions", 0) or 0) > 0


def record_computed_blocks(
    request: Any,
    blocks: Any,
    connector_name: str | None = None,
) -> None:
    """Record newly materialized local KV/state blocks for byte accounting."""
    if not enabled():
        return
    state = _ensure_state(request, connector_name=connector_name)
    _record_block_ids(
        state.computed_kv_block_ids,
        blocks,
        only_unhashed=False,
    )


def record_remote_requested_blocks(
    request: Any,
    blocks: Any,
    connector_name: str | None = None,
) -> None:
    """Record decode-side blocks allocated for remote KV receive/reuse."""
    if not enabled():
        return
    state = _ensure_state(request, connector_name=connector_name)
    _record_block_ids(
        state.remote_requested_kv_block_ids,
        blocks,
        only_unhashed=True,
    )


def emit_request_summary(
    request: Any,
    *,
    finished: bool,
    connector_name: str | None = None,
    kv_cache_config: Any | None = None,
    block_size: int | None = None,
    tensor_parallel_size: int | None = None,
) -> None:
    if not enabled():
        return
    state = getattr(request, "pd_trace", None)
    if state is None or state.emitted:
        return
    if connector_name and state.connector is None:
        state.connector = connector_name

    now = time.perf_counter()
    if state.kv_compute_start_ts is not None:
        state.timing_ms["kv_compute_wall_ms"] = max(
            0.0, (now - state.kv_compute_start_ts) * 1000.0
        )
    else:
        state.notes.append("missing_kv_compute_start")

    arrival_time = getattr(request, "arrival_time", None)
    if isinstance(arrival_time, (int, float)):
        state.timing_ms["backend_ttft_ms"] = max(
            0.0, (time.time() - float(arrival_time)) * 1000.0
        )

    measured_ms = sum(
        state.timing_ms.get(field, 0.0)
        for field in (
            "queue_before_remote_kv_ms",
            "queue_after_remote_kv_ms",
            "local_kv_lookup_ms",
            "external_kv_lookup_ms",
            "remote_kv_wait_ms",
            "kv_compute_wall_ms",
        )
    )
    backend_ttft_ms = state.timing_ms.get("backend_ttft_ms", 0.0)
    other_backend_ms = backend_ttft_ms - measured_ms
    if other_backend_ms < 0:
        state.notes.append(
            f"negative_other_backend_ms:{other_backend_ms:.3f};clamped_to_zero"
        )
        other_backend_ms = 0.0
    state.timing_ms["other_backend_ms"] = other_backend_ms
    for field in _TIMING_FIELDS:
        state.timing_ms.setdefault(field, 0.0)

    state.flags["finished"] = bool(finished)
    state.flags["preempted"] = int(getattr(request, "num_preemptions", 0) or 0) > 0
    kv_bytes = _build_kv_bytes(
        state,
        kv_cache_config=kv_cache_config,
        tensor_parallel_size=tensor_parallel_size,
    )

    record = {
        "schema_version": 1,
        "event": "vllm_pd_request_summary",
        "instance_id": state.instance_id,
        "configured_role": state.configured_role,
        "observed_role": state.observed_role,
        "pid": os.getpid(),
        "request_id": state.request_id,
        "pd_coordination_id": state.pd_coordination_id,
        "transfer_id": state.transfer_id,
        "kv_connector": state.connector,
        "kv_transfer_params": state.kv_transfer_params,
        "tokens": {
            "prompt_tokens": state.prompt_tokens,
            "local_cached_tokens": state.local_cached_tokens,
            "external_cached_tokens": state.external_cached_tokens,
            "computed_tokens": state.computed_tokens,
        },
        "kv_bytes": kv_bytes,
        "timing_ms": {
            key: round(float(value), 6) for key, value in state.timing_ms.items()
        },
        "flags": state.flags,
        "notes": state.notes,
    }
    if _write_record(record):
        state.emitted = True
        logger.debug(
            "[PD_TRACE] emitted request summary request_id=%s pd_id=%s role=%s",
            state.request_id,
            state.pd_coordination_id,
            state.observed_role,
        )


def _ensure_state(request: Any, connector_name: str | None = None) -> PDTraceState:
    state = getattr(request, "pd_trace", None)
    if state is not None:
        if connector_name and state.connector is None:
            state.connector = connector_name
        return state

    cfg = _get_config()
    params = getattr(request, "kv_transfer_params", None)
    if not isinstance(params, dict):
        params = {}
    transfer_id = _string_or_none(params.get("transfer_id"))
    request_id = str(getattr(request, "request_id", ""))
    state = PDTraceState(
        request_id=request_id,
        instance_id=cfg.instance_id,
        configured_role=cfg.configured_role,
        observed_role=_observed_role(params),
        pd_coordination_id=transfer_id or _normalize_request_id(request_id),
        transfer_id=transfer_id,
        connector=connector_name,
        kv_transfer_params=_sanitize(params),
        prompt_tokens=int(getattr(request, "num_prompt_tokens", 0) or 0),
        timing_ms={field: 0.0 for field in _TIMING_FIELDS},
        flags={
            "load_kv_async": False,
            "chunked_prefill": False,
            "preempted": int(getattr(request, "num_preemptions", 0) or 0) > 0,
            "finished": False,
        },
    )
    setattr(request, "pd_trace", state)
    logger.debug(
        "[PD_TRACE] initialized request trace request_id=%s pd_id=%s role=%s",
        state.request_id,
        state.pd_coordination_id,
        state.observed_role,
    )
    return state


def _observed_role(params: dict[str, Any]) -> str:
    if params.get("do_remote_decode"):
        return "prefill_producer"
    if params.get("do_remote_prefill"):
        return "decode_consumer"
    return "regular"


def _normalize_request_id(request_id: str) -> str:
    for prefix in ("chatcmpl-", "cmpl-", "req-", "gnt-"):
        if request_id.startswith(prefix):
            return request_id[len(prefix) :]
    return request_id


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _add_timing(state: PDTraceState, field: str, value_ms: float) -> None:
    state.timing_ms[field] = state.timing_ms.get(field, 0.0) + max(0.0, value_ms)


def _record_block_ids(
    target: dict[int, set[int]],
    blocks: Any,
    *,
    only_unhashed: bool,
) -> None:
    groups = getattr(blocks, "blocks", None)
    if groups is None:
        groups = blocks
    if groups is None:
        return
    for group_index, group in enumerate(groups):
        group_ids = target.setdefault(group_index, set())
        for block in group:
            if isinstance(block, int):
                block_id = block
                is_null = block_id < 0
                block_hash = None
            else:
                is_null = bool(getattr(block, "is_null", False))
                block_id = getattr(block, "block_id", None)
                block_hash = getattr(block, "block_hash", None)
            if is_null or block_id is None:
                continue
            if only_unhashed and block_hash is not None:
                continue
            try:
                parsed = int(block_id)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                group_ids.add(parsed)


def _build_kv_bytes(
    state: PDTraceState,
    *,
    kv_cache_config: Any | None,
    tensor_parallel_size: int | None,
) -> dict[str, Any]:
    method = "kv_cache_group_materialized_blocks_v1"
    result: dict[str, Any] = {
        "computed_kv_bytes_estimated": 0,
        "computed_kv_bytes_per_rank_estimated": 0,
        "remote_requested_kv_bytes_estimated": 0,
        "remote_requested_kv_bytes_per_rank_estimated": 0,
        "producer_tensor_parallel_size": 1,
        "scope": "fleet_physical",
        "method": method,
        "block_count_source": "scheduler_new_materialized_block_ids",
        "groups": [],
    }
    if kv_cache_config is None:
        _append_note_once(state, "kv_bytes_estimate_unavailable:missing_kv_cache_config")
        return result

    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not groups:
        _append_note_once(state, "kv_bytes_estimate_unavailable:missing_kv_cache_groups")
        return result

    tp_size = _parse_positive_int(tensor_parallel_size, default=1)
    if tp_size <= 0:
        _append_note_once(state, "kv_bytes_estimate_unavailable:invalid_tp_size")
        tp_size = 1
    result["producer_tensor_parallel_size"] = tp_size

    computed_per_rank = 0
    remote_requested_per_rank = 0
    group_results: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        group_info = _group_byte_info(group, index, state)
        if group_info is None:
            return result
        computed_blocks = len(state.computed_kv_block_ids.get(index, set()))
        remote_requested_blocks = len(
            state.remote_requested_kv_block_ids.get(index, set())
        )
        group_block_bytes_per_rank = int(group_info["group_block_bytes_per_rank"])
        computed_group_per_rank = computed_blocks * group_block_bytes_per_rank
        remote_group_per_rank = remote_requested_blocks * group_block_bytes_per_rank
        computed_per_rank += computed_group_per_rank
        remote_requested_per_rank += remote_group_per_rank
        group_results.append(
            {
                **group_info,
                "new_materialized_blocks": computed_blocks,
                "remote_requested_blocks": remote_requested_blocks,
                "computed_kv_bytes_per_rank_estimated": computed_group_per_rank,
                "remote_requested_kv_bytes_per_rank_estimated": remote_group_per_rank,
            }
        )

    result["groups"] = group_results
    result["computed_kv_bytes_per_rank_estimated"] = computed_per_rank
    result["computed_kv_bytes_estimated"] = computed_per_rank * tp_size
    result["remote_requested_kv_bytes_per_rank_estimated"] = remote_requested_per_rank
    result["remote_requested_kv_bytes_estimated"] = remote_requested_per_rank * tp_size

    if state.computed_tokens > 0 and computed_per_rank == 0:
        _append_note_once(state, "missing_materialized_block_counts")

    return result


def _group_byte_info(
    group: Any,
    index: int,
    state: PDTraceState,
) -> dict[str, Any] | None:
    layer_names = getattr(group, "layer_names", None)
    layer_count = len(layer_names) if layer_names is not None else 0
    spec = getattr(group, "kv_cache_spec", None)
    try:
        page_size_bytes = int(getattr(spec, "page_size_bytes"))
    except (TypeError, ValueError, AttributeError, NotImplementedError) as exc:
        _append_note_once(
            state,
            "kv_bytes_estimate_unavailable:"
            f"group_{index}_missing_page_size_bytes:{type(exc).__name__}",
        )
        return None
    if page_size_bytes <= 0:
        _append_note_once(
            state,
            "kv_bytes_estimate_unavailable:"
            f"group_{index}_invalid_page_size",
        )
        return None

    effective_block_size = _parse_positive_int(
        getattr(spec, "block_size", None),
        default=0,
    )
    page_size_is_group_aggregated = hasattr(spec, "kv_cache_specs")
    if page_size_is_group_aggregated:
        group_block_bytes_per_rank = page_size_bytes
        page_size_bytes_per_layer: int | None = None
        byte_source = "spec_page_size_bytes_aggregated"
    else:
        if layer_count <= 0:
            _append_note_once(
                state,
                "kv_bytes_estimate_unavailable:"
                f"group_{index}_missing_layers",
            )
            return None
        group_block_bytes_per_rank = page_size_bytes * layer_count
        page_size_bytes_per_layer = page_size_bytes
        byte_source = "spec_page_size_bytes_per_layer"
    if group_block_bytes_per_rank <= 0:
        _append_note_once(
            state,
            "kv_bytes_estimate_unavailable:"
            f"group_{index}_zero_group_block_bytes",
        )
        return None
    return {
        "group_index": index,
        "kind": type(spec).__name__ if spec is not None else "unknown",
        "layer_count": layer_count,
        "effective_block_size": effective_block_size,
        "page_size_bytes_per_layer": page_size_bytes_per_layer,
        "page_size_bytes_per_group": page_size_bytes,
        "group_block_bytes_per_rank": group_block_bytes_per_rank,
        "byte_source": byte_source,
    }


def _parse_positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed


def _infer_block_size(groups: Any) -> int | None:
    # Kept for downstream tests/imports that may have reached into pd_trace.py
    # while the trace schema still exposed a single block_size field.
    for group in groups:
        spec = getattr(group, "kv_cache_spec", None)
        value = getattr(spec, "block_size", None)
        if value is None:
            continue
        parsed = _parse_positive_int(value, default=0)
        if parsed > 0:
            return parsed
    return None


def _append_note_once(state: PDTraceState, note: str) -> None:
    if note not in state.notes:
        state.notes.append(note)


def _json_number(value: float) -> int | float:
    if not math.isfinite(float(value)):
        return 0.0
    if float(value).is_integer():
        return int(value)
    return float(value)


def _sanitize(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return "<max_depth>"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str in _REDACTED_PARAM_KEYS:
                result[key_str] = "<redacted>"
            else:
                result[key_str] = _sanitize(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(v, depth + 1) for v in value]
    if isinstance(value, str):
        if len(value) > 512:
            return value[:512] + "...<truncated>"
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return repr(value)


def _get_config() -> _TraceConfig:
    global _CONFIG
    if _CONFIG is None:
        enabled_env = os.environ.get("VLLM_PD_TRACE_ENABLE", "").strip().lower()
        enabled_value = enabled_env in _TRUE_VALUES
        path = os.environ.get("VLLM_PD_TRACE_PATH")
        if enabled_value and not path:
            _disable_once("VLLM_PD_TRACE_PATH is missing")
            enabled_value = False
        _CONFIG = _TraceConfig(
            enabled=enabled_value,
            path=path,
            instance_id=os.environ.get("VLLM_PD_TRACE_INSTANCE_ID", "unknown"),
            configured_role=os.environ.get("VLLM_PD_TRACE_ROLE", "unknown"),
        )
    return _CONFIG


def _write_record(record: dict[str, Any]) -> bool:
    handle = _open_trace_file()
    if handle is None:
        return False
    try:
        handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()
        return True
    except OSError as exc:
        _disable_once(f"failed to write trace record: {exc}")
        return False


def _open_trace_file() -> TextIO | None:
    global _TRACE_FILE
    if not enabled():
        return None
    if _TRACE_FILE is not None:
        return _TRACE_FILE
    cfg = _get_config()
    if not cfg.path:
        _disable_once("VLLM_PD_TRACE_PATH is missing")
        return None
    try:
        path = Path(cfg.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _TRACE_FILE = path.open("a", encoding="utf-8", buffering=1)
        return _TRACE_FILE
    except OSError as exc:
        _disable_once(f"failed to open trace file {cfg.path}: {exc}")
        return None


def _disable_once(reason: str) -> None:
    global _DISABLED_REASON
    if _DISABLED_REASON is None:
        _DISABLED_REASON = reason
        logger.warning("[PD_TRACE] disabled: %s", reason)
