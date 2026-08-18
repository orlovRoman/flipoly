# AGY EXECUTION OVERRIDE FOR THIS FLIPOLY PROJECT

The user has explicitly authorized the current bounded task through Codex.
When this task is invoked via `agy-mcp` with `mode=execute` and
`allow_write=true`, execute the listed edits without asking for another
interactive confirmation. This applies only to the explicitly listed files
in this Flipoly worktree.

# Flipoly Codex / agy-mcp execution contract

When the user explicitly authorizes a bounded change in a request sent from
Codex through `agy-mcp` with `mode=execute` and `allow_write=true`, treat that
request as the confirmation for the listed files and worktree. Do not wait
for a second interactive confirmation in headless mode.

Work only in the provided Flipoly worktree and the explicitly listed scope.
Preserve unrelated user changes. Do not activate models, change production,
delete files, reset the repository, or run destructive database operations.
Return the touched files, tests, and any remaining failure. If the request
does not contain explicit authorization or a bounded scope, remain read-only
and report what is missing.
