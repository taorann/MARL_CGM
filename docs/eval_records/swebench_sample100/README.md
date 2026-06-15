# SWE-bench Sample100 Eval Records

This directory archives two local `runs/tmp` evaluation records for the
SWE-bench sample100 experiments. They are copied out of ignored runtime output
so they can travel with the repository.

## Archived Runs

### `swebench_supervised_continue_on_sandbox_bug_2026-06-02_17-07-06_UTC`

- Context: continuation run after a clean baseline over the sample100 set.
- Input tasks in `round_01`: 47.
- `round_01/results.jsonl`: 39 records.
- `round_01` status counts: `pass=3`, `not_pass=6`, `bug=30`.
- Aggregated clean baseline after the round: `pass=47`, `not_pass=11`,
  `bug=0`.
- Overall summary reported `accuracy=0.8103448275862069` over clean records.
- The process ended with return code `-9`, so the run is an interrupted eval
  record rather than a completed benchmark claim.

### `swebench_sample100_current_agent_2026-06-11_08-36-21_UTC`

- Context: later current-agent run over the same style of 100-task sample.
- Input tasks in `round_01`: 100.
- `round_01/results.jsonl`: 32 records.
- `round_01` status counts: `pass=25`, `not_pass=6`, `bug=1`.
- Aggregated clean baseline after the round: `pass=23`, `not_pass=4`,
  `bug=0`, with `remaining_count=73`.
- Overall summary reported `accuracy=0.8518518518518519` over clean records.
- The process also ended with return code `-9`, so it is an interrupted eval
  record rather than a completed benchmark claim.

## Included Files

Each run keeps the original summary, launch script, run/supervisor logs,
baseline JSONL, round task JSONL, `round_01/results.jsonl`,
`round_01/progress.md`, metadata, and traces.

Machine-local process files such as `*.pid` and empty `*.nohup.log` files were
excluded.
