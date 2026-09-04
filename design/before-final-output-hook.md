# Native BeforeFinalOutput hook

`BeforeFinalOutput` is the common delivery boundary for Claire, Charlie, Erin,
Macey, and future Tau agents. It lives in the Tau agent loop. It is not an
extension and does not require an agent-specific Python integration.

## Runtime contract

Tau invokes the hook only after the provider has completed a tool-free
assistant candidate. While the hook runs, Tau has not emitted `message_start`,
`message_update`, or `message_end` for that candidate and has not persisted it.

Command hooks receive JSON on standard input:

```json
{
  "hook_event_name": "BeforeFinalOutput",
  "session_id": "session-id",
  "cwd": "/agent/root",
  "last_assistant_message": "candidate response"
}
```

They may return a complete replacement:

```json
{
  "hookSpecificOutput": {
    "replacementText": "corrected response"
  }
}
```

`appendText` is also supported, but the built-in deployment probe uses a full
replacement so the change is unmistakable. Neither belongs in the eventual
writing policy.

Hook failures are fail-open: Tau delivers the unchanged candidate. Error and
aborted provider messages bypass replacement.

## Deployment proof

Install the Tau wheel, then merge this entry into the agent's
`.tau/hooks.json`:

```json
{
  "hooks": {
    "BeforeFinalOutput": [
      {
        "hooks": [
          {
            "type": "builtin",
            "name": "final_output_probe"
          }
        ]
      }
    ]
  }
}
```

Run one ordinary prompt that needs no tool call. A successful deployment
replaces the draft with this temporary Spanish response:

```text
PRUEBA: el borrador interno fue reemplazado. TURN_END_HOOK_FIRED
```

Then remove the probe entry. Replace it with the shared correction command only
after its instruction set and behavioral evaluation are ready. The built-in
probe never launches a subprocess; it exists only to verify the packaged source
boundary consistently across deployments.

## Version adoption

The capability first ships in `tau-by-clarity==0.57.0`. Agents with exact Tau
pins must move to that version or later. Agents using a range must refresh their
lock file. A future agent receives the capability automatically from any later
Tau version, but final-output correction remains off until `BeforeFinalOutput`
is configured.
