# AGENTS.md — Agent Execution Protocol for Flipoly

This project follows the strict 3-tier Gate Protocol defined in `GEMINI.md`.

## 1. GATING RULES (MANDATORY)

- **TYPE A: Questions / Diagnostics / Error Analysis**
  - Triggers: words like "why", "how", "what happened", "check", "?", log inspection.
  - Mode: **STRICT READ-ONLY**.
  - Forbidden: Any code edits (`replace_file_content`, `write_to_file`), git operations, DB mutations, container restarts.
  - Action: Perform read-only investigation, provide structured text analysis, and **STOP**. Never execute self-directed fixes.

- **TYPE B: Action Commands**
  - Triggers: Explicit imperative commands: "Fix", "Apply", "Implement", "Write code".
  - Mode: **CODE & MODIFY ACCESS**.
  - Action: Implement requested changes, test, and commit.

- **TYPE C: Production Deploy**
  - Triggers: Strictly requires the explicit user word "Задеплой" (or "Deploy").
  - Mode: **DEPLOYMENT ACCESS**.

## 2. PRODUCTION CONSTRAINTS
- Production server: `agent-gemini-cli-poly`
- Docker compose: `docker compose` (no hyphen)
- Never activate models, modify production data, or restart containers without explicit user authorization.
