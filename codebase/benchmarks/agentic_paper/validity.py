"""Episode validity and the retry policy (benchmark v3). Pure functions, no I/O.

An episode is judged only from recorded evidence, never from answer quality.

    infra_invalid          the provider failed: unrecoverable call error, or EVERY reply was
                           empty AND each empty reply is explained by provider-side evidence
                           (filtered/blocked, no candidate, provider anomaly finish reason)
    unclassified_empty     (almost) every reply was empty, cause NOT established (missing
                           metadata, unknown finish reason). Kept in the data, reported
                           separately, NOT re-queued automatically
    valid_empty_completion every reply came back with a normal completed status
                           (STOP / MAX_TOKENS / native function-call part) but no visible text:
                           genuine model behaviour, counted as an ordinary failure
    valid                  anything else (including wrong, incomplete or declined answers)

Retry policy (run.py): only `infra_invalid` is replaced, up to --max-attempts total attempts;
the original attempt file/row is retained and each replacement row links to it.
`unclassified_empty` is re-queued only with the explicit --retry-unclassified flag.
"""
from .providers_instrumented import PROVIDER_SIDE

EMPTY_FRACTION = 0.8        # share of model calls that must be empty to characterise the episode
MIN_CALLS = 3               # ...over at least this many calls (one stray empty reply is noise)
VALID_EMPTY = {"empty_stop", "thinking_only_stop", "token_limit_empty", "function_call_only",
               "malformed_function_call"}


def classify_episode(prov, termination, n_tool_calls, final_answer):
    """-> (validity, evidence dict). `prov` is InstrumentedEpisodeLLM.summary().

    The replies that DID come back are examined first: an episode whose every reply was empty
    never produced a usable move, and is labelled by WHY (provider-side evidence vs. the model's
    own empty/malformed completions) even if quota errors followed (retries of an identical,
    deterministic prompt amplify into 429s). Only when no such pattern exists does a provider
    failure make the episode infra_invalid."""
    log = prov.get("call_log") or []
    replies = [c for c in log if not c.get("provider_failure")]
    n = len(replies)
    classes = prov.get("empty_reply_classes") or {}
    n_empty = sum(classes.values())
    ev = {"model_calls": len(log), "replies": n, "empty_replies": n_empty, "empty_classes": classes,
          "provider_failure_calls": prov.get("provider_failure_calls", 0)}
    answered = bool((final_answer or "").strip()) or n_tool_calls > 0
    if n >= MIN_CALLS and n_empty / n >= EMPTY_FRACTION and not answered:
        if set(classes) <= PROVIDER_SIDE:
            ev["reason"] = "all replies empty and each explained by provider-side evidence"
            return "infra_invalid", ev
        if set(classes) <= VALID_EMPTY:
            ev["reason"] = ("every reply came back empty with a normal completed status "
                            f"({', '.join(sorted(classes))}): the model produced no usable move")
            return "valid_empty_completion", ev
        ev["reason"] = "replies empty; cause not established from the recorded metadata"
        return "unclassified_empty", ev
    if prov.get("infra_failed"):
        ev["reason"] = f"provider failure: {prov.get('infra_reason')}"
        return "infra_invalid", ev
    if termination == "timeout" and (prov.get("pool_wait_s") or 0) >= 300:
        ev["reason"] = "wall clock exhausted while waiting on provider rate limits"
        return "infra_invalid", ev
    return "valid", ev


def should_retry(validity, attempts, max_attempts, retry_unclassified=False):
    """Explicit retry policy. `attempts` = attempts already made for the cell."""
    if attempts >= max_attempts:
        return False
    if validity == "infra_invalid":
        return True
    return bool(retry_unclassified and validity == "unclassified_empty" and attempts < 2)
