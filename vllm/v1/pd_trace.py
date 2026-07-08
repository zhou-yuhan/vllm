# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
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


def emit_request_summary(
    request: Any,
    *,
    finished: bool,
    connector_name: str | None = None,
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
