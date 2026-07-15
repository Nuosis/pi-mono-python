## Problem
OpenAI Responses streaming with `gpt-5.5` failed on a simple prompt with an un-awaited `AsyncResponses.create` coroutine warning and then `'dict' object has no attribute 'content'`.

## Hypothesis List

| # | Hypothesis | Null Hypothesis | Status |
|---|------------|-----------------|--------|
| 1 | Provider output shape mismatches shared stream processor | OpenAI Responses provider passes an object with `.content` to `process_responses_stream()` | FALSIFIED |
| 2 | SDK stream creation is not awaited | `client.responses.create()` returns a ready async iterator, not an awaitable | FALSIFIED |

## Debug Evidence
`packages/ai/src/pi_ai/providers/openai_responses.py` initialized `output` as a dict, while `packages/ai/src/pi_ai/providers/openai_responses_shared.py` read `output.content` immediately.

Regression tests using `AsyncMock` for `responses.create` verified the SDK call is awaited and the final message streams as typed assistant output.

## Current Hypothesis
Root cause found: the OpenAI/Azure Responses providers were using legacy dict output plus an un-awaited async SDK call while the shared processor expects typed assistant output and an async iterator.

---

## Claire voice post-tool timeout — 2026-07-15

### Problem

The production `claire_ea` voice session on `/google-voice/session2` received
“ask for Anastasia, do you see it?”, completed three read-only action lookups,
then produced neither a final transcript nor audio. The Tau 0.56.13 trace
contained `tau.provider_request` for the post-tool turn but no
`tau.provider_response`.

### Observable success

For every Anthropic-compatible provider turn, including MiniMax, Tau emits one
raw response envelope containing the SDK stream events and final SDK message.
If iteration or parsing fails, it still emits the events received before the
failure. The exact Claire production replay must then show the initial and
post-tool provider responses, successful read-only tool results, persisted
`transcript_out`, and non-empty audio output.

### Hypothesis list

| # | Hypothesis | Null hypothesis | Status |
|---|------------|-----------------|--------|
| 1 | Tau's Anthropic provider never invokes `on_response` | The adapter invokes the configured callback for every provider turn | CONFIRMED; repaired in 0.56.14 candidate |
| 2 | The 90-second gap was provider unavailability | The exact captured MiniMax request can start and complete promptly | FALSIFIED by a direct 3.478-second streamed replay |
| 3 | The tool results lacked the requested record | The read tools returned Anastasia records before the final turn | FALSIFIED by production `voice_events` and tool-result readback |

### Current hypothesis

Tau 0.56.13 wires `AgentSession._on_provider_response` into
`SimpleStreamOptions`, but `pi_ai.providers.anthropic.stream_simple` never calls
it. This is a provider-adapter control-plane defect. Repair the adapter callback
contract first, then replay the unchanged production scenario and use the newly
visible raw response to isolate any remaining tool-choice or stream-lifecycle
failure.

### Repair evidence

- Contract tests first failed with zero callbacks on both completed and broken
  streams, then passed after the adapter change. A third test proves closing a
  partially consumed async stream still emits the response received so far.
- The adjacent provider/stream slice passed 46 tests from workspace source.
- The built 0.56.14 wheel passed its response-instrumentation and stream tests
  from a clean installed environment (9 tests).
- A disposable container using the Claire server's real MiniMax credential
  loaded the candidate wheel directly, completed a MiniMax-M3 turn in 1.273
  seconds, and captured one serializable envelope with nine raw events and a
  final SDK message.

This evidence proves the provider callback contract. It does not yet prove the
Claire voice workflow; that requires publishing, deploying, and replaying the
exact production route and utterance.

### Exact replay after 0.56.14

Production session `voice-claire_ea-622a89a7e99d` on
`/google-voice/session2` repeated the literal utterance and still timed out.
Timestamped logs showed Tau took about 72 seconds to reach the first provider
request. The first MiniMax/tool turn then completed and a second MiniMax request
started, but Clarify's 90-second per-turn deadline expired three seconds into
that final call. MiniMax completed roughly six seconds after the deadline.

A candidate RPC-readiness event was tested but falsified as the cause: the same
production image reached that boundary in 3.33 seconds, and a direct exact Tau
run completed in 9.616 seconds. The candidate was not published or propagated.
That direct run returned a clarification without calling a lookup tool, exposing
a separate reproducible tool-choice failure even when the timeout is absent.

### Post-tool stream boundary and 0.56.16 candidate

Production session `voice-claire_ea-f466adf419d9` executed
`review.list_pending` and `action.list_tasks`, then the post-tool MiniMax stream
emitted no events for 90 seconds. Clarify sent its RPC abort, but Tau could not
acknowledge it while the provider iterator was blocked; the Tau subprocess
remained alive and the caller received no transcript or audio. This locates the
remaining lifecycle defect at the Anthropic-compatible provider read boundary.

The 0.56.16 candidate bounds MiniMax stream inactivity at 30 seconds and retries
the same provider request once only when the stream emitted zero events. It does
not retry after a partial event, so completed or partially streamed output is
not duplicated. Focused evidence:

- 60 Anthropic/provider/stream tests passed, including successful retry,
  no-retry-after-partial-output, and stop-after-one-retry controls.
- 48 AgentSession tests passed on the cumulative line containing the published
  0.56.15 turn-end workflow guards.
- The built 0.56.16 wheel loaded directly in a disposable Claire-server
  container and completed a real MiniMax-M3 turn in 1.310 seconds with one raw
  response callback, nine provider events, and a final SDK message.

The real-provider smoke had `idle_retries=0`; it proves the normal stream path
was preserved, not that a naturally occurring provider stall retried. The exact
Claire production voice replay remains the end-to-end completion gate.
