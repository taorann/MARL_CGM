# GraphPlanner Progress

- total: 92
- pass: 48
- not_pass: 14
- bug: 30
- accuracy: 0.522
- bug_excluded_accuracy: 0.774
- current_run_total: 39
- baseline_total: 53

- baseline_label: supervisor_clean_baseline_round_1
- baseline_counts: `{"bug": 0, "not_pass": 8, "pass": 45}`
- current_run_counts: `{"bug": 30, "not_pass": 6, "pass": 3}`

## django__django-10554

- phase: finished
- final_status: not_pass
- reason: max_steps after 2784.5s
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 106044, "graph_nodes": 48199, "max_steps": 48, "worker": 1}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 9 | memory_commit | ok | 0.0s | committed=1 memory=3 changed=True |
| 10 | repair | test_failed | 238.1s | status=test_failed rolled_back=True paths=django/db/models/query.py |
| 11 | explore_find | ok | 0.2s | results=0 |
| 12 | explore_find | ok | 30.6s | results=12 |
| 13 | read | ok | 2.4s | function django/db/models/lookups.py:160-165 |
| 14 | explore_find | ok | 17.7s | results=12 |
| 15 | read | ok | 2.4s | function django/db/models/sql/compiler.py:465-625 |
| 16 | read | ok | 2.4s | function django/db/models/sql/compiler.py:410-463 |
| 17 | read | ok | 2.4s | function django/db/models/sql/compiler.py:254-371 |
| 18 | memory_commit | ok | 0.0s | committed=1 memory=4 changed=True |
| 19 | repair | infra_retryable | 300.2s | status=infra_retryable rolled_back=None paths= |
| 20 | repair | test_failed | 286.2s | status=test_failed rolled_back=True paths=django/db/models/sql/compiler.py |
| 21 | read | ok | 2.9s | function django/db/models/sql/compiler.py:342-358 |
| 22 | read | ok | 2.4s | function django/db/models/sql/compiler.py:342-358 |
| 23 | memory_commit_note | ok | 0.0s | {"notes": [{"note": "These functions are directly involved in the reported issue where union with ordering breaks upon subsequent order_by calls.", "tag": null}, {"note": "This function handles the creation of combined queries and calls clear_ordering which is relevant to the ordering issue", "tag":...<truncated 529 chars> |
| 24 | read | ok | 2.3s | function django/db/models/sql/compiler.py:342-358 |
| 25 | read | ok | 2.4s | function django/db/models/sql/compiler.py:342-358 |
| 26 | read | ok | 2.4s | function django/db/models/sql/compiler.py:342-358 |
| 27 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 28 | memory_commit | ok | 0.0s | committed=2 memory=6 changed=True |
| 29 | repair | infra_retryable | 300.5s | status=infra_retryable rolled_back=None paths= |
| 30 | repair | infra_retryable | 250.7s | status=infra_retryable rolled_back=None paths= |
| 31 | repair | test_failed | 277.8s | status=test_failed rolled_back=True paths=django/db/models/sql/compiler.py |
| 32 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 33 | memory_commit | ok | 0.0s | committed=1 memory=7 changed=True |
| 34 | repair | infra_retryable | 300.6s | status=infra_retryable rolled_back=None paths= |
| 35 | repair | test_failed | 230.4s | status=test_failed rolled_back=True paths=django/db/models/sql/compiler.py |
| 36 | read | ok | 2.4s | function django/db/models/sql/compiler.py:350-366 |
| 37 | explore_expand | ok | 0.1s | results=1 |
| 38 | read | ok | 2.3s | function django/db/models/sql/compiler.py:350-366 |
| 39 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 40 | read | ok | 2.9s | function django/db/models/sql/compiler.py:337-353 |
| 41 | read | ok | 2.5s | function django/db/models/sql/compiler.py:350-366 |
| 42 | read | ok | 2.4s | function django/db/models/sql/compiler.py:332-348 |
| 43 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 44 | read | ok | 2.5s | function django/db/models/sql/compiler.py:342-358 |
| 45 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 46 | explore_find | ok | 28.8s | results=12 |
| 47 | read | ok | 2.3s | function django/db/models/sql/compiler.py:337-353 |
| 48 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |

## django__django-13964

- phase: finished
- final_status: not_pass
- reason: max_steps after 2709.5s
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 113734, "graph_nodes": 52029, "max_steps": 48, "worker": 2}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 9 | read | ok | 2.5s | function django/db/models/fields/related.py:896-897 |
| 10 | explore_find | ok | 2.6s | results=1 |
| 11 | read | ok | 2.4s | function django/db/models/fields/related.py:643-644 |
| 12 | read | ok | 2.4s | function django/db/models/fields/related.py:631-632 |
| 13 | read | ok | 3.4s | function django/db/models/fields/related.py:611-628 |
| 14 | memory_commit | ok | 0.0s | committed=5 memory=7 changed=True |
| 15 | repair | infra_retryable | 300.7s | status=infra_retryable rolled_back=None paths= |
| 16 | repair | test_failed | 324.4s | status=test_failed rolled_back=True paths=django/db/models/fields/related.py |
| 17 | explore_find | ok | 2.4s | results=1 |
| 18 | read | ok | 2.9s | class django/db/models/fields/related_descriptors.py:82-275 |
| 19 | memory_commit | ok | 0.0s | committed=1 memory=8 changed=True |
| 20 | read | ok | 2.8s | class django/db/models/fields/related_descriptors.py:260-275 |
| 21 | read | ok | 2.4s | function django/db/models/fields/related.py:965-971 |
| 22 | repair | patch_rejected | 237.5s | status=patch_rejected rolled_back=None paths= |
| 23 | read | ok | 2.4s | class django/db/models/fields/related_descriptors.py:82-150 |
| 24 | read | ok | 2.4s | class django/db/models/fields/related_descriptors.py:150-275 |
| 25 | read | ok | 2.9s | class django/db/models/fields/related_descriptors.py:250-265 |
| 26 | read | ok | 2.4s | class django/db/models/fields/related_descriptors.py:82-120 |
| 27 | explore_find | ok | 29.4s | results=12 |
| 28 | read | ok | 2.4s | function django/db/models/fields/related_descriptors.py:203-267 |
| 29 | memory_commit | ok | 0.0s | committed=1 memory=9 changed=True |
| 30 | repair | test_failed | 217.6s | status=test_failed rolled_back=True paths=django/db/models/fields/related.py |
| 31 | explore_find | ok | 28.8s | results=12 |
| 32 | read | blocked | 0.2s | blocked: unknown node_id: func:django/db/models/fields/related.py:ForeignKey.pre_save:973 |
| 33 | explore_find | ok | 29.1s | results=12 |
| 34 | read | ok | 2.8s | function django/db/models/fields/related.py:794-832 |
| 35 | explore_find | ok | 29.6s | results=12 |
| 36 | read | ok | 2.4s | function django/db/models/fields/related.py:965-971 |
| 37 | read | ok | 2.3s | function django/db/models/fields/related.py:429-438 |
| 38 | memory_commit | ok | 0.0s | committed=1 memory=10 changed=True |
| 39 | explore_find | ok | 2.4s | results=1 |
| 40 | read | ok | 2.8s | function django/db/models/fields/mixins.py:12-19 |
| 41 | repair | infra_retryable | 300.5s | status=infra_retryable rolled_back=None paths= |
| 42 | repair | infra_retryable | 300.4s | status=infra_retryable rolled_back=None paths= |
| 43 | repair | infra_retryable | 300.4s | status=infra_retryable rolled_back=None paths= |
| 44 | repair | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 45 | memory_commit | ok | 0.0s | committed=1 memory=11 changed=True |
| 46 | read | ok | 2.9s | function django/db/models/fields/related.py:965-971 |
| 47 | repair | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 48 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |

## astropy__astropy-13453

- phase: finished
- final_status: not_pass
- reason: max_steps after 2107.8s
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 50825, "graph_nodes": 24440, "max_steps": 48, "worker": 3}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 9 | explore_find | ok | 30.3s | results=12 |
| 10 | read | ok | 2.4s | class astropy/io/ascii/html.py:254-477 |
| 11 | read | ok | 3.4s | class astropy/io/ascii/html.py:254-477 |
| 12 | read | ok | 2.4s | class astropy/io/ascii/html.py:254-477 |
| 13 | read | ok | 2.4s | function astropy/io/ascii/html.py:342-456 |
| 14 | read | ok | 2.3s | function astropy/io/ascii/html.py:458-477 |
| 15 | memory_commit | ok | 0.0s | committed=1 memory=2 changed=True |
| 16 | repair | test_failed | 278.6s | status=test_failed rolled_back=True paths=astropy/io/ascii/html.py |
| 17 | explore_find | ok | 11.8s | results=5 |
| 18 | read | ok | 2.3s | function astropy/table/pprint.py:48-137 |
| 19 | memory_commit | ok | 0.0s | committed=1 memory=3 changed=True |
| 20 | repair | test_failed | 282.9s | status=test_failed rolled_back=True paths=astropy/io/ascii/html.py |
| 21 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 22 | explore_find | ok | 28.9s | results=12 |
| 23 | read | ok | 2.4s | function astropy/io/ascii/ui.py:762-813 |
| 24 | explore_find | ok | 34.0s | results=12 |
| 25 | read | ok | 2.3s | function astropy/io/ascii/core.py:934-938 |
| 26 | read | ok | 2.3s | function astropy/io/ascii/ui.py:762-813 |
| 27 | read | ok | 2.4s | function astropy/io/ascii/core.py:934-938 |
| 28 | read | ok | 2.4s | function astropy/io/ascii/ui.py:762-813 |
| 29 | explore_find | ok | 28.2s | results=12 |
| 30 | read | ok | 2.9s | function astropy/io/ascii/core.py:934-938 |
| 31 | memory_commit | ok | 0.0s | committed=1 memory=4 changed=True |
| 32 | explore_find | ok | 29.7s | results=12 |
| 33 | explore_find | ok | 29.8s | results=12 |
| 34 | explore_find | ok | 30.3s | results=12 |
| 35 | repair | test_failed | 268.1s | status=test_failed rolled_back=True paths=astropy/io/ascii/html.py |
| 36 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 37 | explore_find | ok | 23.5s | results=12 |
| 38 | read | ok | 4.2s | class astropy/io/ascii/core.py:61-155 |
| 39 | read | ok | 2.8s | class astropy/io/ascii/core.py:61-155 |
| 40 | read | ok | 2.3s | function astropy/table/pprint.py:48-137 |
| 41 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 42 | memory_commit_note | ok | 0.0s | {"notes": [{"note": "HTML.write method that needs to properly handle formats parameter", "tag": null}, {"note": "HTML.fill_values method shows how column values are processed with fill values, providing insight into where format functions should be applied", "tag": null}, {"note": "Function that pro...<truncated 402 chars> |
| 43 | read | ok | 3.0s | class astropy/io/ascii/core.py:61-155 |
| 44 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 45 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 46 | read | ok | 2.4s | class astropy/io/ascii/core.py:61-155 |
| 47 | memory_commit | ok | 0.0s | committed=1 memory=5 changed=True |
| 48 | read | ok | 2.9s | class astropy/io/ascii/core.py:61-155 |

## django__django-11206

- phase: finished
- final_status: pass
- reason: env_done
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 105352, "graph_nodes": 47897, "max_steps": 48, "worker": 4}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 1 | run_failed_test | failed | 5.5s | test=failed rc=1 frames=0 |
| 2 | explore_find | ok | 1.4s | results=1 |
| 3 | read | ok | 2.8s | function django/utils/numberformat.py:7-79 |
| 4 | memory_commit | ok | 0.0s | committed=1 memory=1 changed=True |
| 5 | repair | infra_retryable | 300.2s | status=infra_retryable rolled_back=None paths= |
| 6 | repair | infra_retryable | 300.2s | status=infra_retryable rolled_back=None paths= |
| 7 | repair | syntax_failed | 449.3s | status=syntax_failed rolled_back=True paths= |
| 8 | repair | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path |
| 9 | repair | passed | 228.1s | status=passed rolled_back=False paths=django/utils/numberformat.py |

## astropy__astropy-13033

- phase: finished
- final_status: not_pass
- reason: max_steps after 2473.2s
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 50533, "graph_nodes": 24225, "max_steps": 48, "worker": 1}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 9 | read | ok | 2.9s | function astropy/timeseries/core.py:57-85 |
| 10 | memory_commit | ok | 0.0s | committed=1 memory=2 changed=True |
| 11 | repair | test_failed | 157.2s | status=test_failed rolled_back=True paths=astropy/timeseries/core.py |
| 12 | explore_find | ok | 0.1s | results=0 |
| 13 | explore_find | ok | 7.8s | results=3 |
| 14 | read | ok | 2.3s | function astropy/table/table.py:2597-2636 |
| 15 | read | ok | 2.3s | function astropy/table/table.py:2638-2685 |
| 16 | memory_commit | ok | 0.0s | committed=2 memory=4 changed=True |
| 17 | repair | test_failed | 247.0s | status=test_failed rolled_back=True paths=astropy/timeseries/core.py |
| 18 | explore_find | ok | 0.1s | results=0 |
| 19 | explore_find | ok | 0.3s | results=0 |
| 20 | read | ok | 2.4s | class astropy/timeseries/sampled.py:18-383 |
| 21 | memory_commit | ok | 0.0s | committed=1 memory=5 changed=True |
| 22 | repair | infra_retryable | 300.2s | status=infra_retryable rolled_back=None paths= |
| 23 | repair | test_failed | 243.3s | status=test_failed rolled_back=True paths=astropy/timeseries/core.py |
| 24 | read | ok | 2.8s | function astropy/timeseries/core.py:57-85 |
| 25 | memory_commit_note | ok | 0.0s | {"notes": [{"note": "Contains the _check_required_columns method which is responsible for validating required columns and generating error messages", "tag": null}, {"note": "Function that contains the buggy error reporting logic for required columns validation", "tag": null}, {"note": "Column remova...<truncated 660 chars> |
| 26 | read | ok | 2.5s | function astropy/timeseries/core.py:57-85 |
| 27 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 28 | explore_find | ok | 0.2s | results=0 |
| 29 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 30 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 31 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 32 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 33 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 34 | explore_expand | ok | 0.0s | results=2 |
| 35 | explore_expand | ok | 0.0s | results=2 |
| 36 | memory_commit_note | ok | 0.0s | {"notes": [{"note": "Contains the _check_required_columns method which is responsible for validating required columns and generating error messages", "tag": null}, {"note": "Function that contains the buggy error reporting logic for required columns validation", "tag": null}, {"note": "Column remova...<truncated 1050 chars> |
| 37 | explore_expand | ok | 0.0s | results=2 |
| 38 | explore_expand | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 39 | memory_commit_note | ok | 0.0s | {"notes": [{"note": "Contains the _check_required_columns method which is responsible for validating required columns and generating error messages", "tag": null}, {"note": "Function that contains the buggy error reporting logic for required columns validation", "tag": null}, {"note": "Column remova...<truncated 1610 chars> |
| 40 | explore_expand | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 41 | read | ok | 2.4s | function astropy/timeseries/core.py:88-92 |
| 42 | memory_commit | ok | 0.0s | committed=1 memory=6 changed=True |
| 43 | repair | infra_retryable | 300.2s | status=infra_retryable rolled_back=None paths= |
| 44 | repair | test_failed | 215.9s | status=test_failed rolled_back=True paths=astropy/timeseries/core.py |
| 45 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 46 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 47 | explore_expand | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 48 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |

## django__django-12308

- phase: finished
- final_status: pass
- reason: env_done
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 109303, "graph_nodes": 50257, "max_steps": 48, "worker": 2}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 1 | run_failed_test | failed | 7.3s | test=failed rc=1 frames=0 |
| 2 | explore_find | ok | 2.5s | results=1 |
| 3 | read | ok | 2.3s | function django/contrib/admin/utils.py:380-402 |
| 4 | memory_commit | ok | 0.0s | committed=1 memory=1 changed=True |
| 5 | explore_find | ok | 11.2s | results=4 |
| 6 | read | ok | 2.4s | class django/forms/fields.py:1226-1278 |
| 7 | memory_commit | ok | 0.0s | committed=1 memory=2 changed=True |
| 8 | repair | test_failed | 200.6s | status=test_failed rolled_back=True paths=django/contrib/admin/utils.py |
| 9 | read | ok | 2.4s | class django/db/models/fields/json.py:16-122 |
| 10 | read | ok | 2.3s | class django/db/models/fields/json.py:16-122 |
| 11 | memory_commit | ok | 0.0s | committed=1 memory=3 changed=True |
| 12 | repair | passed | 150.1s | status=passed rolled_back=False paths=django/contrib/admin/utils.py |

## django__django-16560

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-83912bf6-2a78-944a-a7ff-72db75c3f725","request_id":"83912bf6-2a78-944a-a7ff-72db75c3f725"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 124416, "graph_nodes": 55968, "max_steps": 48, "worker": 3}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 1 | explore_find | ok | 3.4s | results=1 |
| 2 | read | ok | 2.4s | class django/db/models/constraints.py:19-81 |
| 3 | run_failed_test | failed | 7.4s | test=failed rc=1 frames=0 |
| 4 | memory_commit | ok | 0.0s | committed=1 memory=1 changed=True |
| 5 | explore_find | ok | 2.5s | results=1 |
| 6 | explore_find | ok | 2.6s | results=1 |
| 7 | read | ok | 2.4s | class django/db/models/constraints.py:84-144 |
| 8 | read | ok | 2.4s | class django/db/models/constraints.py:156-405 |
| 9 | memory_commit | ok | 0.0s | committed=2 memory=3 changed=True |
| 10 | repair | infra_retryable | 300.3s | status=infra_retryable rolled_back=None paths= |
| 11 | read | ok | 2.4s | class django/db/models/constraints.py:156-405 |
| 12 | repair | infra_retryable | 300.4s | status=infra_retryable rolled_back=None paths= |
| 13 | repair | infra_retryable | 300.3s | status=infra_retryable rolled_back=None paths= |
| 14 | read | ok | 2.4s | class django/db/models/constraints.py:156-405 |
| 15 | repair | infra_retryable | 300.3s | status=infra_retryable rolled_back=None paths= |
| 16 | repair | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path |
| 17 | repair | infra_retryable | 300.3s | status=infra_retryable rolled_back=None paths= |
| 18 | repair | patch_rejected | 250.1s | status=patch_rejected rolled_back=None paths= |
| 19 | repair | infra_retryable | 269.7s | status=infra_retryable rolled_back=None paths= |
| 20 | repair | patch_rejected | 287.4s | status=patch_rejected rolled_back=None paths= |
| 21 | read | ok | 2.4s | class django/db/models/constraints.py:19-81 |
| 22 | read | ok | 2.9s | class django/db/models/constraints.py:19-81 |
| 23 | read | ok | 2.4s | class django/db/models/constraints.py:380-405 |
| 24 | read | ok | 2.5s | function django/db/models/constraints.py:65-66 |
| 25 | memory_commit | ok | 0.0s | committed=1 memory=4 changed=True |
| 26 | repair | syntax_failed | 545.4s | status=syntax_failed rolled_back=True paths= |
| 27 | repair | infra_retryable | 301.5s | status=infra_retryable rolled_back=None paths= |

## django__django-15127

- phase: finished
- final_status: not_pass
- reason: max_steps after 1185.6s
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 118709, "graph_nodes": 53967, "max_steps": 48, "worker": 4}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 9 | read | ok | 2.8s | function django/contrib/messages/utils.py:5-12 |
| 10 | memory_commit | ok | 0.0s | committed=1 memory=3 changed=True |
| 11 | repair | test_failed | 91.5s | status=test_failed rolled_back=True paths=django/contrib/messages/storage/base.py |
| 12 | read | ok | 2.4s | class django/contrib/messages/storage/base.py:1-10 |
| 13 | read | ok | 2.3s | class django/contrib/messages/storage/base.py:1-41 |
| 14 | read | ok | 2.4s | function django/contrib/messages/storage/base.py:40-41 |
| 15 | memory_commit | ok | 0.0s | committed=1 memory=4 changed=True |
| 16 | repair | patch_rejected | 85.8s | status=patch_rejected rolled_back=None paths= |
| 17 | explore_find | ok | 4.9s | results=2 |
| 18 | read | ok | 2.4s | function django/contrib/messages/storage/base.py:19-25 |
| 19 | read | ok | 2.4s | function django/contrib/messages/storage/base.py:40-41 |
| 20 | memory_commit | ok | 0.0s | committed=1 memory=5 changed=True |
| 21 | repair | test_failed | 171.0s | status=test_failed rolled_back=True paths=django/contrib/messages/storage/base.py |
| 22 | read | ok | 2.4s | function django/contrib/messages/storage/base.py:40-41 |
| 23 | explore_find | ok | 29.7s | results=12 |
| 24 | read | ok | 2.4s | class django/conf/__init__.py:50-166 |
| 25 | memory_commit | ok | 0.0s | committed=1 memory=6 changed=True |
| 26 | read | ok | 2.9s | class django/contrib/messages/storage/base.py:1-5 |
| 27 | repair | patch_rejected | 99.3s | status=patch_rejected rolled_back=None paths= |
| 28 | memory_commit_note | ok | 0.0s | {"notes": [{"note": "LEVEL_TAGS is a static dictionary assigned at module load time, and Message.level_tag uses it directly. This doesn't reflect runtime changes from @override_settings.", "tag": null}, {"note": "get_level_tags function provides the correct merged tags from defaults and settings, wh...<truncated 968 chars> |
| 29 | explore_find | ok | 4.9s | results=2 |
| 30 | read | ok | 2.4s | function django/contrib/messages/storage/base.py:40-41 |
| 31 | read | ok | 2.9s | class django/contrib/messages/storage/base.py:1-5 |
| 32 | read | ok | 2.9s | assignment django/contrib/messages/storage/base.py:1-10 |
| 33 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 34 | read | ok | 2.9s | assignment django/contrib/messages/storage/base.py:1-10 |
| 35 | explore_find | ok | 5.0s | results=2 |
| 36 | read | ok | 2.9s | assignment django/contrib/messages/storage/base.py:4-34 |
| 37 | explore_find | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 38 | memory_delete | ok | 0.0s | deleted=1 memory=5 changed=True |
| 39 | memory_commit | ok | 0.0s | committed=1 memory=6 changed=True |
| 40 | read | ok | 2.6s | class django/conf/__init__.py:97-106 |
| 41 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 42 | read | ok | 2.9s | class django/contrib/messages/storage/base.py:1-41 |
| 43 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 44 | explore_find | ok | 0.3s | results=0 |
| 45 | explore_find | ok | 0.2s | results=0 |
| 46 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 47 | read | ok | 2.4s | assignment django/contrib/messages/storage/base.py:1-10 |
| 48 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |

## django__django-13344

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-2399cf5d-fb7d-9064-b2fd-c0f079bc4835","request_id":"2399cf5d-fb7d-9064-b2fd-c0f079bc4835"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 111602, "graph_nodes": 51218, "max_steps": 48, "worker": 1}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 1 | explore_find | ok | 0.3s | results=12 |
| 2 | explore_find | ok | 2.9s | results=1 |
| 3 | explore_expand | ok | 0.1s | results=11 |
| 4 | explore_find | ok | 2.6s | results=1 |
| 5 | read | ok | 2.4s | class django/core/handlers/asgi.py:126-287 |
| 6 | run_failed_test | failed | 79.5s | test=failed rc=1 frames=0 |
| 7 | memory_commit | ok | 0.0s | committed=1 memory=1 changed=True |
| 8 | explore_find | ok | 2.5s | results=1 |
| 9 | read | ok | 2.4s | function django/core/handlers/base.py:26-95 |
| 10 | read | ok | 2.4s | function django/core/handlers/base.py:210-273 |
| 11 | explore_find | ok | 8.2s | results=3 |
| 12 | memory_commit | blocked | 0.0s | blocked: memory_commit requires an explicit read before commit: func:django/core/handlers/exception.py:convert_exception_to_response:21. explore_find previews and expand candidates are orientation context, not repair evidence. |
| 13 | read | ok | 2.4s | function django/core/handlers/exception.py:21-51 |
| 14 | memory_commit | ok | 0.0s | committed=3 memory=4 changed=True |
| 15 | repair | infra_retryable | 300.2s | status=infra_retryable rolled_back=None paths= |
| 16 | repair | infra_retryable | 300.3s | status=infra_retryable rolled_back=None paths= |
| 17 | repair | infra_retryable | 300.3s | status=infra_retryable rolled_back=None paths= |
| 18 | repair | infra_retryable | 300.2s | status=infra_retryable rolled_back=None paths= |
| 19 | repair | test_failed | 317.1s | status=test_failed rolled_back=True paths=django/core/handlers/base.py |
| 20 | explore_find | ok | 3.2s | results=1 |
| 21 | read | ok | 2.4s | function django/core/handlers/base.py:97-122 |
| 22 | memory_commit | ok | 0.0s | committed=1 memory=5 changed=True |
| 23 | repair | infra_retryable | 300.3s | status=infra_retryable rolled_back=None paths= |
| 24 | repair | test_failed | 299.5s | status=test_failed rolled_back=True paths=django/core/handlers/base.py |
| 25 | explore_find | ok | 30.5s | results=12 |
| 26 | read | ok | 2.3s | function django/core/handlers/base.py:80-96 |
| 27 | read | ok | 2.4s | function django/core/handlers/base.py:70-100 |
| 28 | read | ok | 2.9s | function django/core/handlers/base.py:80-95 |
| 29 | read | ok | 2.4s | function django/core/handlers/base.py:35-96 |
| 30 | explore_find | ok | 3.0s | results=1 |
| 31 | read | ok | 2.4s | class django/utils/deprecation.py:88-147 |
| 32 | memory_commit | ok | 0.0s | committed=1 memory=6 changed=True |
| 33 | read | ok | 3.4s | function django/core/handlers/base.py:35-96 |
| 34 | repair | infra_retryable | 1.5s | status=infra_retryable rolled_back=None paths= |
| 35 | repair | infra_retryable | 300.4s | status=infra_retryable rolled_back=None paths= |

## django__django-12406

- phase: finished
- final_status: not_pass
- reason: max_steps after 1814.2s
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 108052, "graph_nodes": 49623, "max_steps": 48, "worker": 2}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 9 | memory_commit | ok | 0.0s | committed=1 memory=2 changed=True |
| 10 | repair | test_failed | 261.4s | status=test_failed rolled_back=True paths=django/forms/widgets.py |
| 11 | memory_delete | ok | 0.0s | deleted=1 memory=1 changed=True |
| 12 | memory_commit | ok | 0.0s | committed=1 memory=2 changed=True |
| 13 | explore_find | ok | 17.0s | results=7 |
| 14 | read | ok | 2.4s | class django/forms/models.py:279-467 |
| 15 | memory_commit | ok | 0.0s | committed=1 memory=3 changed=True |
| 16 | repair | test_failed | 162.9s | status=test_failed rolled_back=True paths=django/forms/widgets.py |
| 17 | explore_find | ok | 31.3s | results=12 |
| 18 | read | ok | 2.8s | function django/db/models/fields/related.py:973-983 |
| 19 | explore_find | ok | 2.5s | results=1 |
| 20 | read | ok | 2.9s | class django/forms/models.py:1174-1284 |
| 21 | memory_commit | ok | 0.0s | committed=2 memory=5 changed=True |
| 22 | repair | test_failed | 254.9s | status=test_failed rolled_back=True paths=django/forms/models.py |
| 23 | read | ok | 2.4s | function django/db/models/fields/related.py:973-983 |
| 24 | read | ok | 2.4s | function django/db/models/fields/related.py:973-983 |
| 25 | read | ok | 2.4s | class django/forms/models.py:1174-1284 |
| 26 | explore_find | ok | 31.9s | results=12 |
| 27 | read | ok | 2.4s | function django/db/models/fields/related.py:973-983 |
| 28 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 29 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 30 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 31 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 32 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 33 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 34 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 35 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 36 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 37 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 38 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 39 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 40 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 41 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 42 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 43 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 44 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 45 | read | ok | 2.5s | function django/db/models/fields/related.py:973-983 |
| 46 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 47 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 48 | read | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |

## django__django-15503

- phase: finished
- final_status: pass
- reason: env_done
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 120756, "graph_nodes": 54683, "max_steps": 48, "worker": 3}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 1 | run_failed_test | failed | 7.9s | test=failed rc=1 frames=0 |
| 2 | explore_find | ok | 7.1s | results=3 |
| 3 | read | ok | 2.4s | class django/db/models/fields/json.py:16-126 |
| 4 | memory_commit | ok | 0.0s | committed=1 memory=1 changed=True |
| 5 | explore_find | ok | 28.7s | results=12 |
| 6 | explore_find | ok | 32.7s | results=12 |
| 7 | explore_expand | ok | 0.1s | results=12 |
| 8 | explore_find | ok | 28.4s | results=12 |
| 9 | read | ok | 2.9s | function django/db/models/fields/json.py:175-206 |
| 10 | read | ok | 2.9s | function django/db/models/fields/json.py:129-139 |
| 11 | memory_commit | ok | 0.0s | committed=2 memory=3 changed=True |
| 12 | repair | test_failed | 168.5s | status=test_failed rolled_back=True paths=django/db/models/fields/json.py |
| 13 | read | ok | 2.4s | function django/db/models/fields/json.py:229-232 |
| 14 | explore_find | ok | 10.1s | results=4 |
| 15 | read | ok | 2.9s | class django/db/models/fields/json.py:241-247 |
| 16 | memory_commit | ok | 0.0s | committed=2 memory=5 changed=True |
| 17 | repair | test_failed | 436.8s | status=test_failed rolled_back=True paths=django/db/models/fields/json.py |
| 18 | read | ok | 2.4s | function django/db/models/fields/json.py:208-211 |
| 19 | memory_commit | ok | 0.0s | committed=1 memory=6 changed=True |
| 20 | repair | passed | 205.1s | status=passed rolled_back=False paths=django/db/models/fields/json.py |

## django__django-11265

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-a47720d7-3969-9fd0-98eb-1c23dd91eb01","request_id":"a47720d7-3969-9fd0-98eb-1c23dd91eb01"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 105650, "graph_nodes": 48052, "max_steps": 48, "worker": 4}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 1 | run_failed_test | failed | 7.8s | test=failed rc=1 frames=0 |
| 2 | explore_find | ok | 3.0s | results=1 |
| 3 | read | ok | 2.4s | function django/db/models/sql/query.py:1645-1713 |
| 4 | memory_commit | ok | 0.0s | committed=1 memory=1 changed=True |
| 5 | explore_find | ok | 31.8s | results=12 |
| 6 | explore_find | ok | 30.3s | results=12 |
| 7 | explore_find | ok | 29.4s | results=12 |
| 8 | explore_find | ok | 0.1s | results=0 |
| 9 | explore_find | ok | 30.3s | results=12 |
| 10 | explore_find | ok | 31.4s | results=12 |
| 11 | repair | patch_rejected | 106.5s | status=patch_rejected rolled_back=None paths= |
| 12 | explore_find | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 13 | explore_find | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 14 | explore_find | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 15 | repair | error | 95.3s | error: RemoteSweError remote rollback failed rc=1: runner_exception: OSError(7, 'Argument list too long') |
| 16 | repair | error | 94.6s | error: RemoteSweError remote rollback failed rc=1: runner_exception: OSError(7, 'Argument list too long') |
| 17 | repair | patch_rejected | 67.5s | status=patch_rejected rolled_back=None paths= |
| 18 | explore_find | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 19 | explore_find | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 20 | repair | patch_rejected | 56.1s | status=patch_rejected rolled_back=None paths= |
| 21 | explore_find | blocked | 0.0s | blocked: repeated identical action exceeded limit 3; choose a different implementation node or commit/repair path; hydrated read code exists in W outside repair memory M, so commit causal read nodes before repair |
| 22 | explore_find | ok | 31.0s | results=12 |
| 23 | read | ok | 2.5s | function django/db/models/sql/query.py:271-275 |
| 24 | memory_commit | ok | 0.0s | committed=1 memory=2 changed=True |
| 25 | repair | patch_rejected | 98.6s | status=patch_rejected rolled_back=None paths= |

## django__django-12965

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-7f53abc9-57f8-9dce-a82c-8183de39885d","request_id":"7f53abc9-57f8-9dce-a82c-8183de39885d"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 109578, "graph_nodes": 50400, "max_steps": 48, "worker": 1}`

| step | tool | status | elapsed | summary |
| ---: | --- | --- | ---: | --- |
| 1 | run_failed_test | failed | 8.4s | test=failed rc=1 frames=0 |
| 2 | explore_find | ok | 32.2s | results=12 |
| 3 | read | ok | 2.3s | function django/db/models/query.py:724-751 |
| 4 | read | ok | 2.3s | function django/db/models/deletion.py:379-447 |
| 5 | memory_commit | ok | 0.0s | committed=2 memory=2 changed=True |
| 6 | read | ok | 2.4s | function django/db/models/deletion.py:165-199 |
| 7 | memory_commit | ok | 0.0s | committed=1 memory=3 changed=True |
| 8 | read | ok | 2.9s | function django/db/models/query.py:756-767 |
| 9 | memory_commit | ok | 0.0s | committed=1 memory=4 changed=True |
| 10 | repair | test_failed | 196.5s | status=test_failed rolled_back=True paths=django/db/models/deletion.py |

## astropy__astropy-14369

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-e31d741a-4a16-99bd-88c3-b0a00f887465","request_id":"e31d741a-4a16-99bd-88c3-b0a00f887465"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 51556, "graph_nodes": 24787, "max_steps": 48, "worker": 2}`

## django__django-15554

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-6a240523-0b99-9587-93fa-72de2ae848cc","request_id":"6a240523-0b99-9587-93fa-72de2ae848cc"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 120910, "graph_nodes": 54743, "max_steps": 48, "worker": 3}`

## django__django-12273

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-b1d2312e-a015-93d4-940f-a1f0b896697b","request_id":"b1d2312e-a015-93d4-940f-a1f0b896697b"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 109245, "graph_nodes": 49518, "max_steps": 48, "worker": 4}`

## django__django-12663

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-5dcf1dee-16c2-9116-9f12-8a4689046ae2","request_id":"5dcf1dee-16c2-9116-9f12-8a4689046ae2"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 108897, "graph_nodes": 50031, "max_steps": 48, "worker": 1}`

## django__django-11276

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-fe2bfd34-9a8f-939b-87e4-f14650b17f83","request_id":"fe2bfd34-9a8f-939b-87e4-f14650b17f83"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 105472, "graph_nodes": 47943, "max_steps": 48, "worker": 2}`

## django__django-11820

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-2e1a2e5d-c5db-90e7-a0de-3266150a74cd","request_id":"2e1a2e5d-c5db-90e7-a0de-3266150a74cd"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 107514, "graph_nodes": 48927, "max_steps": 48, "worker": 3}`

## astropy__astropy-13236

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-7512f8e9-bc50-9373-8b08-9bebf61c2a01","request_id":"7512f8e9-bc50-9373-8b08-9bebf61c2a01"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 50693, "graph_nodes": 24361, "max_steps": 48, "worker": 4}`

## django__django-15375

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-7d9c8a52-3695-99bf-b109-7c006e94b1de","request_id":"7d9c8a52-3695-99bf-b109-7c006e94b1de"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 119837, "graph_nodes": 54404, "max_steps": 48, "worker": 1}`

## django__django-16315

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-f5c2b89e-ea77-933a-9584-716ea2adf0b4","request_id":"f5c2b89e-ea77-933a-9584-716ea2adf0b4"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 123896, "graph_nodes": 55820, "max_steps": 48, "worker": 2}`

## django__django-15037

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-efb8ad27-17fb-9cab-947a-19e2203195f3","request_id":"efb8ad27-17fb-9cab-947a-19e2203195f3"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 118404, "graph_nodes": 53833, "max_steps": 48, "worker": 3}`

## django__django-13658

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-0f554585-af8d-9c68-97a4-e0f1810bea38","request_id":"0f554585-af8d-9c68-97a4-e0f1810bea38"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 112834, "graph_nodes": 51704, "max_steps": 48, "worker": 4}`

## django__django-16145

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-8bc52531-3348-94bf-aad4-0d3b25ed049f","request_id":"8bc52531-3348-94bf-aad4-0d3b25ed049f"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 123365, "graph_nodes": 55604, "max_steps": 48, "worker": 1}`

## django__django-11400

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-16f52c36-4711-9f52-8e22-f0d46347de29","request_id":"16f52c36-4711-9f52-8e22-f0d46347de29"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 106819, "graph_nodes": 48609, "max_steps": 48, "worker": 2}`

## django__django-10880

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-93153baa-2b5f-9e1a-ac3b-a9b2ed9d8b8a","request_id":"93153baa-2b5f-9e1a-ac3b-a9b2ed9d8b8a"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 104571, "graph_nodes": 47774, "max_steps": 48, "worker": 3}`

## django__django-13028

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-a87d2c58-e2fd-9e20-ab50-63fdf84fec62","request_id":"a87d2c58-e2fd-9e20-ab50-63fdf84fec62"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 109920, "graph_nodes": 50560, "max_steps": 48, "worker": 4}`

## django__django-14315

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-604df5a7-8a4d-946f-bdec-3a5ffc992609","request_id":"604df5a7-8a4d-946f-bdec-3a5ffc992609"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 115111, "graph_nodes": 52570, "max_steps": 48, "worker": 1}`

## django__django-15499

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-050baa36-5ad4-9ad8-a197-1e037d171a7c","request_id":"050baa36-5ad4-9ad8-a197-1e037d171a7c"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 120744, "graph_nodes": 54680, "max_steps": 48, "worker": 2}`

## django__django-14311

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-cdaaa7a0-7609-94dd-9aab-e5ef9635e8f8","request_id":"cdaaa7a0-7609-94dd-9aab-e5ef9635e8f8"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 115409, "graph_nodes": 52665, "max_steps": 48, "worker": 3}`

## django__django-10914

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-dcd376e6-7d8f-95ed-81c9-e998f9fc1b31","request_id":"dcd376e6-7d8f-95ed-81c9-e998f9fc1b31"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 104459, "graph_nodes": 47613, "max_steps": 48, "worker": 4}`

## django__django-14434

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-e5b9142b-174d-947d-a629-f16e12bd1568","request_id":"e5b9142b-174d-947d-a629-f16e12bd1568"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 115347, "graph_nodes": 52655, "max_steps": 48, "worker": 1}`

## django__django-11292

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-2d1497d5-acbc-928a-89de-374380542c2f","request_id":"2d1497d5-acbc-928a-89de-374380542c2f"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 105497, "graph_nodes": 47948, "max_steps": 48, "worker": 2}`

## django__django-12039

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-66f1a2e4-6ee0-9592-8055-37abdec4d405","request_id":"66f1a2e4-6ee0-9592-8055-37abdec4d405"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 108117, "graph_nodes": 49162, "max_steps": 48, "worker": 3}`

## django__django-14376

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-57baf3d9-d438-99ab-b9b5-8346fca667f1","request_id":"57baf3d9-d438-99ab-b9b5-8346fca667f1"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 115208, "graph_nodes": 52598, "max_steps": 48, "worker": 4}`

## django__django-16116

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-47937e7a-6e42-97e6-8a25-c65c9c2fad23","request_id":"47937e7a-6e42-97e6-8a25-c65c9c2fad23"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 123339, "graph_nodes": 55600, "max_steps": 48, "worker": 1}`

## django__django-16454

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-67fff68d-2496-971d-9881-18e5ecefba45","request_id":"67fff68d-2496-971d-9881-18e5ecefba45"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 124093, "graph_nodes": 55882, "max_steps": 48, "worker": 2}`

## django__django-7530

- phase: finished
- final_status: bug
- reason: planner output parse failed: planner HTTP 400: {"error":{"message":"Access denied, please make sure your account is in good standing. For details, see: https://help.aliyun.com/zh/model-studio/error-code#overdue-payment","type":"Arrearage","param":null,"code":"Arrearage"},"id":"chatcmpl-436d559d-651a-9142-bcc9-dc766706c667","request_id":"436d559d-651a-9142-bcc9-dc766706c667"}
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 96778, "graph_nodes": 43705, "max_steps": 48, "worker": 3}`

## django__django-11333

- phase: graph_built
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 106335, "graph_nodes": 48330, "max_steps": 48, "worker": 4}`

## django__django-15098

- phase: graph_built
- metadata: `{"backend": "remote_swe", "cgm": "http", "graph_edges": 118887, "graph_nodes": 54048, "max_steps": 48, "worker": 1}`

## django__django-14771

- phase: building_graph
- metadata: `{"backend": "remote_swe", "cgm": "http", "max_steps": 48, "worker": 2}`

## django__django-16899

- phase: building_graph
- metadata: `{"backend": "remote_swe", "cgm": "http", "max_steps": 48, "worker": 3}`
