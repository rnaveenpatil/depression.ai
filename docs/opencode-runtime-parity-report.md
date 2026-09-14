# OpenCode-style runtime parity report

## Implemented in this change

- **Canonical conversation state:** `ContextManager` is now the live source of truth used by the execution loop. `AgentLoop` keeps execution telemetry instead of maintaining a second persistent message history.
- **Tool-call/result integrity:** assistant tool calls and their `tool` results retain matching IDs and are treated as an atomic context group.
- **Provider message adaptation:** canonical context messages are converted to the provider `Message` format only at the model boundary.
- **History trimming:** recent-context selection does not intentionally split an assistant tool-call group from its tool results.
- **Orphan protection:** invalid/orphan tool results and incomplete tool-call groups are removed before model context is built.
- **Duplicate-turn protection:** repeated user/assistant persistence of the same turn is deduplicated by `ContextManager`.
- **Compaction safety:** context is normalized before and after compaction so broken tool-call relationships are not carried forward.
- **Tool execution feedback:** the existing permission → registry → execution path now feeds the exact tool result back into canonical model context.
- **Final response:** after an execution loop reaches its stopping condition, the final response is generated from canonical context rather than selecting an arbitrary older assistant message.
- **Regression coverage:** tests were added for atomic tool-call/result history and orphan-result removal.

## Runtime shape

```text
Session
  -> ContextManager
  -> LLM
  -> Permission gate
  -> ToolRegistry / executor
  -> Tool result
  -> ContextManager
  -> LLM
  -> ...
  -> final response
```

## What is deliberately not claimed as complete OpenCode parity

This change aligns the **core context/tool execution path** with the OpenCode-style architecture. It does not claim byte-for-byte or feature-for-feature parity with the OpenCode product. Remaining work includes persistent child-session subagents, fully checkpoint-based compaction, richer process/background execution, provider-aware token budgeting, and complete planner/task verification semantics.

## Verification

The repository now contains focused regression tests for the highest-risk context invariant. The GitHub content API was used to update the `main` branch; a full local test run was not available through the GitHub connector, so runtime verification should still be performed locally with the project's test suite before treating this as production-ready.
