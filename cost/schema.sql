-- Idempotent: every script that touches the DB runs this first.

CREATE TABLE IF NOT EXISTS requests (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  ts            INTEGER NOT NULL,                        -- unix seconds
  model         TEXT    NOT NULL,
  tier          TEXT    NOT NULL,                        -- claude | local-long
  input_tok     INTEGER NOT NULL DEFAULT 0,
  output_tok    INTEGER NOT NULL DEFAULT 0,
  actual_cost   REAL    NOT NULL DEFAULT 0.0,            -- USD; 0 for local
  shadow_cost   REAL    NOT NULL DEFAULT 0.0,            -- USD; what Claude would have charged
  latency_ms    INTEGER NOT NULL DEFAULT 0,
  route_reason  TEXT    NOT NULL DEFAULT '',
  task_id       TEXT,                                    -- 16-char SHA256 of <task> for Cline traffic; NULL otherwise
  task_text     TEXT                                     -- truncated <task> body, displayed on /tasks
);

CREATE INDEX IF NOT EXISTS idx_requests_ts      ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_tier    ON requests(tier);
CREATE INDEX IF NOT EXISTS idx_requests_task_id ON requests(task_id);

CREATE TABLE IF NOT EXISTS comparisons (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  ts             INTEGER NOT NULL,
  prompt         TEXT    NOT NULL,
  local_model    TEXT    NOT NULL,
  claude_model   TEXT    NOT NULL,
  local_output   TEXT    NOT NULL DEFAULT '',
  claude_output  TEXT    NOT NULL DEFAULT '',
  local_in_tok   INTEGER NOT NULL DEFAULT 0,
  local_out_tok  INTEGER NOT NULL DEFAULT 0,
  claude_in_tok  INTEGER NOT NULL DEFAULT 0,
  claude_out_tok INTEGER NOT NULL DEFAULT 0,
  local_cost     REAL    NOT NULL DEFAULT 0.0,
  claude_cost    REAL    NOT NULL DEFAULT 0.0,
  local_ms       INTEGER NOT NULL DEFAULT 0,
  claude_ms      INTEGER NOT NULL DEFAULT 0,
  judge_score    REAL    NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_comparisons_ts ON comparisons(ts);
