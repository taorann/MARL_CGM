
## planner_step

```json
{
  "action": {
    "params": {
      "find_type": "class",
      "path_glob": "**/*.py",
      "query": "Case"
    },
    "tool": "explore_find"
  },
  "result": {
    "path_glob": "**/*.py",
    "result_policy": "function/class/method/assignment results include a small implementation preview in W for orientation only; read the node before memory_commit or repair evidence. file-level results list top symbols instead of full file text",
    "results": [
      {
        "code": "1319: class Case(SQLiteNumericMixin, Expression):\n1320:     \"\"\"\n1321:     An SQL searched CASE expression:\n1322: \n1323:         CASE\n1324:             WHEN n > 0\n1325:                 THEN 'positive'\n1326:             WHEN n < 0\n1327:                 THEN 'negative'\n1328:             ELSE 'zero'\n1329:         END\n1330:     \"\"\"\n1331: \n1332:     template = \"CASE %(cases)s ELSE %(default)s END\"\n1333:     case_joiner = \" \"\n1334: \n1335:     def __init__(self, *cases, default=None, output_field=None, **extra):\n1336:         if not all(isinstance(case, When) for case in cases):\n1337:             raise TypeError(\"Positional arguments must all be When objects.\")\n1338:         super().__init__(output_field)\n1339:         self.cases = list(cases)\n1340:         self.default = self._parse_expressions(default)[0]\n1341:         self.extra = extra\n1342: \n1343:     def __str__(self):\n1344:         return \"CASE %s, ELSE %r\" % (\n1345:             \", \".join(str(c) for c in self.cases),\n1346:             self.default,\n1347:         )\n1348: ",
        "code_preview_lines": [
          1319,
          1348
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/db/models/expressions.py:Case:1319",
        "kind": "class",
        "lines": [
          1319,
          1409
        ],
        "local_symbol_references": [
          {
            "appearance_index": 11,
            "id": "class:django/db/models/expressions.py:SQLiteNumericMixin:21",
            "kind": "class",
            "lines": [
              21,
              34
            ],
            "name": "SQLiteNumericMixin",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "SQLiteNumericMixin"
          },
          {
            "appearance_index": 31,
            "id": "class:django/db/models/expressions.py:Expression:462",
            "kind": "class",
            "lines": [
              462,
              490
            ],
            "name": "Expression",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "Expression"
          },
          {
            "appearance_index": 379,
            "id": "func:django/db/models/expressions.py:BaseExpression.output_field:293",
            "kind": "function",
            "lines": [
              293,
              299
            ],
            "name": "BaseExpression.output_field",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "output_field"
          },
          {
            "appearance_index": 444,
            "id": "class:django/db/models/expressions.py:When:1245",
            "kind": "class",
            "lines": [
              1245,
              1315
            ],
            "name": "When",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "When"
          },
          {
            "appearance_index": 648,
            "id": "func:django/db/models/expressions.py:BaseExpression._parse_expressions:200",
            "kind": "function",
            "lines": [
              200,
              206
            ],
            "name": "BaseExpression._parse_expressions",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_parse_expressions"
          }
        ],
        "name": "Case",
        "path": "django/db/models/expressions.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 11,
            "id": "class:django/db/models/expressions.py:SQLiteNumericMixin:21",
            "kind": "class",
            "lines": [
              21,
              34
            ],
            "name": "SQLiteNumericMixin",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "SQLiteNumericMixin"
          },
          {
            "appearance_index": 31,
            "id": "class:django/db/models/expressions.py:Expression:462",
            "kind": "class",
            "lines": [
              462,
              490
            ],
            "name": "Expression",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "Expression"
          },
          {
            "appearance_index": 379,
            "id": "func:django/db/models/expressions.py:BaseExpression.output_field:293",
            "kind": "function",
            "lines": [
              293,
              299
            ],
            "name": "BaseExpression.output_field",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "output_field"
          },
          {
            "appearance_index": 444,
            "id": "class:django/db/models/expressions.py:When:1245",
            "kind": "class",
            "lines": [
              1245,
              1315
            ],
            "name": "When",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "When"
          },
          {
            "appearance_index": 648,
            "id": "func:django/db/models/expressions.py:BaseExpression._parse_expressions:200",
            "kind": "function",
            "lines": [
              200,
              206
            ],
            "name": "BaseExpression._parse_expressions",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_parse_expressions"
          }
        ]
      },
      {
        "code": "  19: class AdminSeleniumTestCase(SeleniumTestCase, StaticLiveServerTestCase):\n  20: \n  21:     available_apps = [\n  22:         \"django.contrib.admin\",\n  23:         \"django.contrib.auth\",\n  24:         \"django.contrib.contenttypes\",\n  25:         \"django.contrib.sessions\",\n  26:         \"django.contrib.sites\",\n  27:     ]\n  28: \n  29:     def wait_until(self, callback, timeout=10):\n  30:         \"\"\"\n  31:         Block the execution of the tests until the specified callback returns a\n  32:         value that is not falsy. This method can be called, for example, after\n  33:         clicking a link or submitting a form. See the other public methods that\n  34:         call this function for more details.\n  35:         \"\"\"\n  36:         from selenium.webdriver.support.wait import WebDriverWait\n  37: \n  38:         WebDriverWait(self.selenium, timeout).until(callback)\n  39: \n  40:     def wait_for_and_switch_to_popup(self, num_windows=2, timeout=10):\n  41:         \"\"\"\n  42:         Block until `num_windows` are present and are ready (usually 2, but can\n  43:         be overridden in the case of pop-ups opening other pop-ups). Switch the\n  44:         current window to the new pop-up.\n  45:         \"\"\"\n  46:         self.wait_until(lambda d: len(d.window_handles) == num_windows, timeout)",
        "code_preview_lines": [
          19,
          46
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/admin/tests.py:AdminSeleniumTestCase:19",
        "kind": "class",
        "lines": [
          19,
          234
        ],
        "local_symbol_references": [
          {
            "appearance_index": 281,
            "id": "func:django/contrib/admin/tests.py:AdminSeleniumTestCase.wait_until:29",
            "kind": "function",
            "lines": [
              29,
              38
            ],
            "name": "AdminSeleniumTestCase.wait_until",
            "path": "django/contrib/admin/tests.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "wait_until"
          },
          {
            "appearance_index": 767,
            "id": "func:django/contrib/admin/tests.py:AdminSeleniumTestCase.wait_for_and_switch_to_popup:40",
            "kind": "function",
            "lines": [
              40,
              48
            ],
            "name": "AdminSeleniumTestCase.wait_for_and_switch_to_popup",
            "path": "django/contrib/admin/tests.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "wait_for_and_switch_to_popup"
          }
        ],
        "name": "AdminSeleniumTestCase",
        "path": "django/contrib/admin/tests.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 281,
            "id": "func:django/contrib/admin/tests.py:AdminSeleniumTestCase.wait_until:29",
            "kind": "function",
            "lines": [
              29,
              38
            ],
            "name": "AdminSeleniumTestCase.wait_until",
            "path": "django/contrib/admin/tests.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "wait_until"
          },
          {
            "appearance_index": 767,
            "id": "func:django/contrib/admin/tests.py:AdminSeleniumTestCase.wait_for_and_switch_to_popup:40",
            "kind": "function",
            "lines": [
              40,
              48
            ],
            "name": "AdminSeleniumTestCase.wait_for_and_switch_to_popup",
            "path": "django/contrib/admin/tests.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "wait_for_and_switch_to_popup"
          }
        ]
      },
      {
        "code": " 287: class CaseInsensitiveMapping(Mapping):\n 288:     \"\"\"\n 289:     Mapping allowing case-insensitive key lookups. Original case of keys is\n 290:     preserved for iteration and string representation.\n 291: \n 292:     Example::\n 293: \n 294:         >>> ci_map = CaseInsensitiveMapping({'name': 'Jane'})\n 295:         >>> ci_map['Name']\n 296:         Jane\n 297:         >>> ci_map['NAME']\n 298:         Jane\n 299:         >>> ci_map['name']\n 300:         Jane\n 301:         >>> ci_map  # original case preserved\n 302:         {'name': 'Jane'}\n 303:     \"\"\"\n 304: \n 305:     def __init__(self, data):\n 306:         self._store = {k.lower(): (k, v) for k, v in self._unpack_items(data)}\n 307: \n 308:     def __getitem__(self, key):\n 309:         return self._store[key.lower()][1]\n 310: \n 311:     def __len__(self):\n 312:         return len(self._store)\n 313: \n 314:     def __eq__(self, other):\n 315:         return isinstance(other, Mapping) and {\n 316:             k.lower(): v for k, v in self.items()",
        "code_preview_lines": [
          287,
          316
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/utils/datastructures.py:CaseInsensitiveMapping:287",
        "kind": "class",
        "lines": [
          287,
          346
        ],
        "local_symbol_references": [
          {
            "appearance_index": 544,
            "id": "func:django/utils/datastructures.py:CaseInsensitiveMapping._unpack_items:329",
            "kind": "function",
            "lines": [
              329,
              346
            ],
            "name": "CaseInsensitiveMapping._unpack_items",
            "path": "django/utils/datastructures.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_unpack_items"
          },
          {
            "appearance_index": 817,
            "id": "func:django/utils/datastructures.py:MultiValueDict.items:177",
            "kind": "function",
            "lines": [
              177,
              183
            ],
            "name": "MultiValueDict.items",
            "path": "django/utils/datastructures.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "items"
          }
        ],
        "name": "CaseInsensitiveMapping",
        "path": "django/utils/datastructures.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 544,
            "id": "func:django/utils/datastructures.py:CaseInsensitiveMapping._unpack_items:329",
            "kind": "function",
            "lines": [
              329,
              346
            ],
            "name": "CaseInsensitiveMapping._unpack_items",
            "path": "django/utils/datastructures.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_unpack_items"
          },
          {
            "appearance_index": 817,
            "id": "func:django/utils/datastructures.py:MultiValueDict.items:177",
            "kind": "function",
            "lines": [
              177,
              183
            ],
            "name": "MultiValueDict.items",
            "path": "django/utils/datastructures.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "items"
          }
        ]
      },
      {
        "code": " 262: class CaseInsensitiveMixin:\n 263:     \"\"\"\n 264:     Mixin to allow case-insensitive comparison of JSON values on MySQL.\n 265:     MySQL handles strings used in JSON context using the utf8mb4_bin collation.\n 266:     Because utf8mb4_bin is a binary collation, comparison of JSON values is\n 267:     case-sensitive.\n 268:     \"\"\"\n 269: \n 270:     def process_lhs(self, compiler, connection):\n 271:         lhs, lhs_params = super().process_lhs(compiler, connection)\n 272:         if connection.vendor == \"mysql\":\n 273:             return \"LOWER(%s)\" % lhs, lhs_params\n 274:         return lhs, lhs_params\n 275: \n 276:     def process_rhs(self, compiler, connection):\n 277:         rhs, rhs_params = super().process_rhs(compiler, connection)\n 278:         if connection.vendor == \"mysql\":\n 279:             return \"LOWER(%s)\" % rhs, rhs_params\n 280:         return rhs, rhs_params",
        "code_preview_lines": [
          262,
          280
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": false,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/db/models/fields/json.py:CaseInsensitiveMixin:262",
        "kind": "class",
        "lines": [
          262,
          280
        ],
        "local_symbol_references": [
          {
            "appearance_index": 301,
            "id": "func:django/db/models/fields/json.py:CaseInsensitiveMixin.process_lhs:270",
            "kind": "function",
            "lines": [
              270,
              274
            ],
            "name": "CaseInsensitiveMixin.process_lhs",
            "path": "django/db/models/fields/json.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "process_lhs"
          },
          {
            "appearance_index": 540,
            "id": "func:django/db/models/fields/json.py:CaseInsensitiveMixin.process_rhs:276",
            "kind": "function",
            "lines": [
              276,
              280
            ],
            "name": "CaseInsensitiveMixin.process_rhs",
            "path": "django/db/models/fields/json.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "process_rhs"
          },
          {
            "appearance_index": 540,
            "id": "func:django/db/models/fields/json.py:JSONExact.process_rhs:286",
            "kind": "function",
            "lines": [
              286,
              294
            ],
            "name": "JSONExact.process_rhs",
            "path": "django/db/models/fields/json.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "process_rhs"
          },
          {
            "appearance_index": 540,
            "id": "func:django/db/models/fields/json.py:KeyTransformExact.process_rhs:447",
            "kind": "function",
            "lines": [
              447,
              469
            ],
            "name": "KeyTransformExact.process_rhs",
            "path": "django/db/models/fields/json.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "process_rhs"
          },
          {
            "appearance_index": 540,
            "id": "func:django/db/models/fields/json.py:KeyTransformNumericLookupMixin.process_rhs:529",
            "kind": "function",
            "lines": [
              529,
              533
            ],
            "name": "KeyTransformNumericLookupMixin.process_rhs",
            "path": "django/db/models/fields/json.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "process_rhs"
          }
        ],
        "name": "CaseInsensitiveMixin",
        "path": "django/db/models/fields/json.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 301,
            "id": "func:django/db/models/fields/json.py:CaseInsensitiveMixin.process_lhs:270",
            "kind": "function",
            "lines": [
              270,
              274
            ],
            "name": "CaseInsensitiveMixin.process_lhs",
            "path": "django/db/models/fields/json.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "process_lhs"
          },
          {
            "appearance_index": 540,
            "id": "func:django/db/models/fields/json.py:CaseInsensitiveMixin.process_rhs:276",
            "kind": "function",
            "lines": [
              276,
              280
            ],
            "name": "CaseInsensitiveMixin.process_rhs",
            "path": "django/db/models/fields/json.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "process_rhs"
          },
          {
            "appearance_index": 540,
            "id": "func:django/db/models/fields/json.py:JSONExact.process_rhs:286",
            "kind": "function",
            "lines": [
              286,
              294
            ],
            "name": "JSONExact.process_rhs",
            "path": "django/db/models/fields/json.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "process_rhs"
          },
          {
            "appearance_index": 540,
            "id": "func:django/db/models/fields/json.py:KeyTransformExact.process_rhs:447",
            "kind": "function",
            "lines": [
              447,
              469
            ],
            "name": "KeyTransformExact.process_rhs",
            "path": "django/db/models/fields/json.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "process_rhs"
          },
          {
            "appearance_index": 540,
            "id": "func:django/db/models/fields/json.py:KeyTransformNumericLookupMixin.process_rhs:529",
            "kind": "function",
            "lines": [
              529,
              533
            ],
            "name": "KeyTransformNumericLookupMixin.process_rhs",
            "path": "django/db/models/fields/json.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "process_rhs"
          }
        ]
      },
      {
        "code": "   5: class StaticLiveServerTestCase(LiveServerTestCase):\n   6:     \"\"\"\n   7:     Extend django.test.LiveServerTestCase to transparently overlay at test\n   8:     execution-time the assets provided by the staticfiles app finders. This\n   9:     means you don't need to run collectstatic before or as a part of your tests\n  10:     setup.\n  11:     \"\"\"\n  12: \n  13:     static_handler = StaticFilesHandler",
        "code_preview_lines": [
          5,
          13
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": false,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/staticfiles/testing.py:StaticLiveServerTestCase:5",
        "kind": "class",
        "lines": [
          5,
          13
        ],
        "local_symbol_references": [],
        "name": "StaticLiveServerTestCase",
        "path": "django/contrib/staticfiles/testing.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": []
      }
    ],
    "tool": "explore_find",
    "warning": null
  },
  "step": 1,
  "visible_thinking": "The issue describes a crash in Django's `Case()` expression when used with `~Q(pk__in=[])`. The error is a SQL syntax error where the generated SQL has an incomplete `CASE WHEN` clause (`CASE WHEN THEN True ELSE False END`). This suggests that the `When` condition is not being properly converted to SQL when the `Q` object is negated with an empty list.\nFirst, I need to understand the components involved:\n1. `Case` and `When` expressions in Django ORM\n2. `Q` objects and their negation\n3. How these are compiled to SQL\nSince I don't have any code read yet, I should start by finding the relevant files/classes. I'll look for the `Case` class, `When` class, and `Q` class in the Django codebase.\nLet's start by finding the `Case` class."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "node_id": "class:django/db/models/expressions.py:Case:1319",
      "view": "body"
    },
    "tool": "read"
  },
  "result": {
    "code": "1319: class Case(SQLiteNumericMixin, Expression):\n1320:     \"\"\"\n1321:     An SQL searched CASE expression:\n1322: \n1323:         CASE\n1324:             WHEN n > 0\n1325:                 THEN 'positive'\n1326:             WHEN n < 0\n1327:                 THEN 'negative'\n1328:             ELSE 'zero'\n1329:         END\n1330:     \"\"\"\n1331: \n1332:     template = \"CASE %(cases)s ELSE %(default)s END\"\n1333:     case_joiner = \" \"\n1334: \n1335:     def __init__(self, *cases, default=None, output_field=None, **extra):\n1336:         if not all(isinstance(case, When) for case in cases):\n1337:             raise TypeError(\"Positional arguments must all be When objects.\")\n1338:         super().__init__(output_field)\n1339:         self.cases = list(cases)\n1340:         self.default = self._parse_expressions(default)[0]\n1341:         self.extra = extra\n1342: \n1343:     def __str__(self):\n1344:         return \"CASE %s, ELSE %r\" % (\n1345:             \", \".join(str(c) for c in self.cases),\n1346:             self.default,\n1347:         )\n1348: \n1349:     def __repr__(self):\n1350:         return \"<%s: %s>\" % (self.__class__.__name__, self)\n1351: \n1352:     def get_source_expressions(self):\n1353:         return self.cases + [self.default]\n1354: \n1355:     def set_source_expressions(self, exprs):\n1356:         *self.cases, self.default = exprs\n1357: \n1358:     def resolve_expression(\n1359:         self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False\n1360:     ):\n1361:         c = self.copy()\n1362:         c.is_summary = summarize\n1363:         for pos, case in enumerate(c.cases):\n1364:             c.cases[pos] = case.resolve_expression(\n1365:                 query, allow_joins, reuse, summarize, for_save\n1366:             )\n1367:         c.default = c.default.resolve_expression(\n1368:             query, allow_joins, reuse, summarize, for_save\n1369:         )\n1370:         return c\n1371: \n1372:     def copy(self):\n1373:         c = super().copy()\n1374:         c.cases = c.cases[:]\n1375:         return c\n1376: \n1377:     def as_sql(\n1378:         self, compiler, connection, template=None, case_joiner=None, **extra_context\n1379:     ):\n1380:         connection.ops.check_expression_support(self)\n1381:         if not self.cases:\n1382:             return compiler.compile(self.default)\n1383:         template_params = {**self.extra, **extra_context}\n1384:         case_parts = []\n1385:         sql_params = []\n1386:         for case in self.cases:\n1387:             try:\n1388:                 case_sql, case_params = compiler.compile(case)\n1389:             except EmptyResultSet:\n1390:                 continue\n1391:             case_parts.append(case_sql)\n1392:             sql_params.extend(case_params)\n1393:         default_sql, default_params = compiler.compile(self.default)\n1394:         if not case_parts:\n1395:             return default_sql, default_params\n1396:         case_joiner = case_joiner or self.case_joiner\n1397:         template_params[\"cases\"] = case_joiner.join(case_parts)\n1398:         template_params[\"default\"] = default_sql\n1399:         sql_params.extend(default_params)\n1400:         template = template or template_params.get(\"template\", self.template)\n1401:         sql = template % template_params\n1402:         if self._output_field_or_none is not None:\n1403:             sql = connection.ops.unification_cast_sql(self.output_field) % sql\n1404:         return sql, sql_params\n1405: \n1406:     def get_group_by_cols(self, alias=None):\n1407:         if not self.cases:\n1408:             return self.default.get_group_by_cols(alias)\n1409:         return super().get_group_by_cols(alias)",
    "dispatch_relationship_context": [],
    "dispatch_tables": [],
    "local_symbol_references": [
      {
        "appearance_index": 11,
        "id": "class:django/db/models/expressions.py:SQLiteNumericMixin:21",
        "kind": "class",
        "lines": [
          21,
          34
        ],
        "name": "SQLiteNumericMixin",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "SQLiteNumericMixin"
      },
      {
        "appearance_index": 31,
        "id": "class:django/db/models/expressions.py:Expression:462",
        "kind": "class",
        "lines": [
          462,
          490
        ],
        "name": "Expression",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "Expression"
      },
      {
        "appearance_index": 379,
        "id": "func:django/db/models/expressions.py:BaseExpression.output_field:293",
        "kind": "function",
        "lines": [
          293,
          299
        ],
        "name": "BaseExpression.output_field",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "output_field"
      },
      {
        "appearance_index": 444,
        "id": "class:django/db/models/expressions.py:When:1245",
        "kind": "class",
        "lines": [
          1245,
          1315
        ],
        "name": "When",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "When"
      },
      {
        "appearance_index": 648,
        "id": "func:django/db/models/expressions.py:BaseExpression._parse_expressions:200",
        "kind": "function",
        "lines": [
          200,
          206
        ],
        "name": "BaseExpression._parse_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "_parse_expressions"
      },
      {
        "appearance_index": 949,
        "id": "func:django/db/models/expressions.py:BaseExpression.get_source_expressions:194",
        "kind": "function",
        "lines": [
          194,
          195
        ],
        "name": "BaseExpression.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      },
      {
        "appearance_index": 949,
        "id": "func:django/db/models/expressions.py:CombinedExpression.get_source_expressions:641",
        "kind": "function",
        "lines": [
          641,
          642
        ],
        "name": "CombinedExpression.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      },
      {
        "appearance_index": 949,
        "id": "func:django/db/models/expressions.py:Func.get_source_expressions:908",
        "kind": "function",
        "lines": [
          908,
          909
        ],
        "name": "Func.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      }
    ],
    "node": {
      "id": "class:django/db/models/expressions.py:Case:1319",
      "kind": "class",
      "lines": [
        1319,
        1409
      ],
      "name": "Case",
      "path": "django/db/models/expressions.py"
    },
    "relationship_context_policy": "consumer candidates are auto-added to W as orientation code only; read the exact node before memory_commit/repair evidence",
    "tool": "read",
    "unread_local_symbol_references": [
      {
        "appearance_index": 11,
        "id": "class:django/db/models/expressions.py:SQLiteNumericMixin:21",
        "kind": "class",
        "lines": [
          21,
          34
        ],
        "name": "SQLiteNumericMixin",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "SQLiteNumericMixin"
      },
      {
        "appearance_index": 31,
        "id": "class:django/db/models/expressions.py:Expression:462",
        "kind": "class",
        "lines": [
          462,
          490
        ],
        "name": "Expression",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "Expression"
      },
      {
        "appearance_index": 379,
        "id": "func:django/db/models/expressions.py:BaseExpression.output_field:293",
        "kind": "function",
        "lines": [
          293,
          299
        ],
        "name": "BaseExpression.output_field",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "output_field"
      },
      {
        "appearance_index": 444,
        "id": "class:django/db/models/expressions.py:When:1245",
        "kind": "class",
        "lines": [
          1245,
          1315
        ],
        "name": "When",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "When"
      },
      {
        "appearance_index": 648,
        "id": "func:django/db/models/expressions.py:BaseExpression._parse_expressions:200",
        "kind": "function",
        "lines": [
          200,
          206
        ],
        "name": "BaseExpression._parse_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "_parse_expressions"
      },
      {
        "appearance_index": 949,
        "id": "func:django/db/models/expressions.py:BaseExpression.get_source_expressions:194",
        "kind": "function",
        "lines": [
          194,
          195
        ],
        "name": "BaseExpression.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      },
      {
        "appearance_index": 949,
        "id": "func:django/db/models/expressions.py:CombinedExpression.get_source_expressions:641",
        "kind": "function",
        "lines": [
          641,
          642
        ],
        "name": "CombinedExpression.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      },
      {
        "appearance_index": 949,
        "id": "func:django/db/models/expressions.py:Func.get_source_expressions:908",
        "kind": "function",
        "lines": [
          908,
          909
        ],
        "name": "Func.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      }
    ],
    "value_flow_context": [
      {
        "argument_to_parameter": [
          {
            "argument": "condition",
            "parameter": "default",
            "position": 0
          }
        ],
        "call": "Case(condition)",
        "call_line": 22,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/aggregates.py:Aggregate.as_sql:103",
          "kind": "function",
          "lines": [
            103,
            128
          ],
          "name": "Aggregate.as_sql",
          "path": "django/db/models/aggregates.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:Case:1319",
          "kind": "class",
          "lines": [
            1319,
            1409
          ],
          "name": "Case",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "output_field",
          "*cases",
          "**extra"
        ]
      },
      {
        "argument_to_parameter": [
          {
            "argument": "When(self.expression, then=True)",
            "parameter": "default",
            "position": 0
          },
          {
            "argument": "False",
            "parameter": "default",
            "position": 1
          }
        ],
        "call": "Case(When(self.expression, then=True), default=False)",
        "call_line": 8,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/expressions.py:OrderBy.as_oracle:1577",
          "kind": "function",
          "lines": [
            1577,
            1589
          ],
          "name": "OrderBy.as_oracle",
          "path": "django/db/models/expressions.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:Case:1319",
          "kind": "class",
          "lines": [
            1319,
            1409
          ],
          "name": "Case",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "output_field",
          "*cases",
          "**extra"
        ]
      },
      {
        "argument_to_parameter": [
          {
            "argument": "When(expr, then=True)",
            "parameter": "default",
            "position": 0
          },
          {
            "argument": "False",
            "parameter": "default",
            "position": 1
          }
        ],
        "call": "Case(When(expr, then=True), default=False)",
        "call_line": 8,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/lookups.py:Lookup.as_oracle:137",
          "kind": "function",
          "lines": [
            137,
            148
          ],
          "name": "Lookup.as_oracle",
          "path": "django/db/models/lookups.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:Case:1319",
          "kind": "class",
          "lines": [
            1319,
            1409
          ],
          "name": "Case",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "output_field",
          "*cases",
          "**extra"
        ]
      },
      {
        "argument_to_parameter": [
          {
            "argument": "*when_statements",
            "parameter": "default",
            "position": 0
          },
          {
            "argument": "field",
            "parameter": "output_field",
            "position": 1
          }
        ],
        "call": "Case(*when_statements, output_field=field)",
        "call_line": 41,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/query.py:QuerySet.bulk_update:855",
          "kind": "function",
          "lines": [
            855,
            905
          ],
          "name": "QuerySet.bulk_update",
          "path": "django/db/models/query.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:Case:1319",
          "kind": "class",
          "lines": [
            1319,
            1409
          ],
          "name": "Case",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "*cases",
          "**extra"
        ]
      }
    ],
    "value_flow_context_policy": "value_flow_context is best-effort implementation evidence from call expressions and signatures; it shows upstream/downstream argument-to-parameter flow but is not a proof of runtime values. Related caller/callee candidates are added to W; read exact nodes before memory_commit/repair."
  },
  "step": 2,
  "visible_thinking": "The user is reporting a bug where using `~Q(pk__in=[])` inside a `Case()` expression causes a syntax error in the generated SQL. The error message shows that the SQL being generated is missing the condition between `WHEN` and `THEN`, resulting in `CASE WHEN THEN True ELSE False END`.\nI have already found the `Case` class in `django/db/models/expressions.py`. I need to understand how the `Case` class processes its `When` conditions, especially when the condition is a negated empty queryset lookup like `~Q(pk__in=[])`.\nFirst, I should read the full `Case` class implementation to understand its structure. Then, I should look at the `When` class, since that's what holds the condition. The problem likely lies in how the condition is being evaluated or converted to SQL when it's a negated empty queryset lookup.\nLet me start by reading the `Case` class in detail."
}
```

## planner_step

```json
{
  "action": {
    "params": {},
    "tool": "run_failed_test"
  },
  "result": {
    "test": {
      "command": "<official SWE-bench eval command omitted; contains benchmark harness setup/test patch>",
      "command_omitted_for_benchmark_hygiene": true,
      "excerpt": "Actual runtime failure summary (benchmark expected values omitted):\nexception_types: sqlite3.OperationalError, django.db.utils.OperationalError\nactual_messages:\n- near \"THEN\": syntax error",
      "implementation_frames": [],
      "parser_error": "missing version for SWE-bench log parser",
      "resolved": null,
      "returncode": 1,
      "runtime_observations": {
        "actual_assertion_values": [],
        "actual_messages": [
          "near \"THEN\": syntax error"
        ],
        "exception_types": [
          "sqlite3.OperationalError",
          "django.db.utils.OperationalError"
        ],
        "failed_selectors": [],
        "omitted_hidden_expected_values": false,
        "policy": "actual runtime output only; benchmark test source and hidden expected values are omitted"
      },
      "status": "failed",
      "tests_status": {}
    },
    "tool": "run_failed_test"
  },
  "step": 3,
  "visible_thinking": "The issue involves a crash in the `Case()` expression when used with `~Q(pk__in=[])`. The error indicates a malformed SQL generation where the `WHEN` clause is empty, leading to a syntax error near `THEN`.\nFrom the code read, I see the `Case` class in `django/db/models/expressions.py`. The problem likely lies in how the `When` conditions are processed, especially when dealing with empty lists in `Q` objects.\nBefore proceeding with any repair, I need to collect the actual runtime behavior by running the failed test. This will provide the necessary `failure_seen` evidence to build a proper repair case."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "note": "The Case class is directly involved in the SQL generation issue where the WHEN clause is malformed.",
      "select_ids": [
        "class:django/db/models/expressions.py:Case:1319"
      ]
    },
    "tool": "memory_commit"
  },
  "result": {
    "already_present_ids": [],
    "committed": [
      "class:django/db/models/expressions.py:Case:1319"
    ],
    "dropped_by_keep_ids": [],
    "explicitly_selected_ids": [
      "class:django/db/models/expressions.py:Case:1319"
    ],
    "memory": [
      {
        "has_code": true,
        "id": "class:django/db/models/expressions.py:Case:1319",
        "kind": "class",
        "lines": [
          1319,
          1409
        ],
        "name": "Case",
        "path": "django/db/models/expressions.py"
      }
    ],
    "memory_changed": true,
    "newly_added_ids": [
      "class:django/db/models/expressions.py:Case:1319"
    ],
    "tool": "memory_commit"
  },
  "step": 4,
  "visible_thinking": "The issue involves a `Case()` expression crashing when used with `~Q(pk__in=[])`, producing a SQL syntax error due to a malformed `WHEN` clause. The error message indicates that the SQL generated is missing a condition between `WHEN` and `THEN`. I have already identified and read the `Case` class from `django/db/models/expressions.py`. Now, I need to commit this node to the repair memory since it's directly involved in the issue. This will allow me to proceed with proposing a repair."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "node_id": "class:django/db/models/expressions.py:Case:1319",
      "view": "file_window:1377-1404"
    },
    "tool": "read"
  },
  "result": {
    "code": "1377:     def as_sql(\n1378:         self, compiler, connection, template=None, case_joiner=None, **extra_context\n1379:     ):\n1380:         connection.ops.check_expression_support(self)\n1381:         if not self.cases:\n1382:             return compiler.compile(self.default)\n1383:         template_params = {**self.extra, **extra_context}\n1384:         case_parts = []\n1385:         sql_params = []\n1386:         for case in self.cases:\n1387:             try:\n1388:                 case_sql, case_params = compiler.compile(case)\n1389:             except EmptyResultSet:\n1390:                 continue\n1391:             case_parts.append(case_sql)\n1392:             sql_params.extend(case_params)\n1393:         default_sql, default_params = compiler.compile(self.default)\n1394:         if not case_parts:\n1395:             return default_sql, default_params\n1396:         case_joiner = case_joiner or self.case_joiner\n1397:         template_params[\"cases\"] = case_joiner.join(case_parts)\n1398:         template_params[\"default\"] = default_sql\n1399:         sql_params.extend(default_params)\n1400:         template = template or template_params.get(\"template\", self.template)\n1401:         sql = template % template_params\n1402:         if self._output_field_or_none is not None:\n1403:             sql = connection.ops.unification_cast_sql(self.output_field) % sql\n1404:         return sql, sql_params",
    "dispatch_relationship_context": [],
    "dispatch_tables": [],
    "local_symbol_references": [
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:BaseExpression.as_sql:208",
        "kind": "function",
        "lines": [
          208,
          234
        ],
        "name": "BaseExpression.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:CombinedExpression.as_sql:664",
        "kind": "function",
        "lines": [
          664,
          676
        ],
        "name": "CombinedExpression.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:DurationExpression.as_sql:738",
        "kind": "function",
        "lines": [
          738,
          753
        ],
        "name": "DurationExpression.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:TemporalSubtraction.as_sql:783",
        "kind": "function",
        "lines": [
          783,
          789
        ],
        "name": "TemporalSubtraction.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:ResolvedOuterRef.as_sql:837",
        "kind": "function",
        "lines": [
          837,
          841
        ],
        "name": "ResolvedOuterRef.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:Func.as_sql:925",
        "kind": "function",
        "lines": [
          925,
          960
        ],
        "name": "Func.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:Value.as_sql:992",
        "kind": "function",
        "lines": [
          992,
          1008
        ],
        "name": "Value.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:RawSQL.as_sql:1059",
        "kind": "function",
        "lines": [
          1059,
          1060
        ],
        "name": "RawSQL.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      }
    ],
    "node": {
      "id": "class:django/db/models/expressions.py:Case:1319",
      "kind": "class",
      "lines": [
        1377,
        1404
      ],
      "name": "Case",
      "path": "django/db/models/expressions.py"
    },
    "relationship_context_policy": "consumer candidates are auto-added to W as orientation code only; read the exact node before memory_commit/repair evidence",
    "tool": "read",
    "unread_local_symbol_references": [
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:BaseExpression.as_sql:208",
        "kind": "function",
        "lines": [
          208,
          234
        ],
        "name": "BaseExpression.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:CombinedExpression.as_sql:664",
        "kind": "function",
        "lines": [
          664,
          676
        ],
        "name": "CombinedExpression.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:DurationExpression.as_sql:738",
        "kind": "function",
        "lines": [
          738,
          753
        ],
        "name": "DurationExpression.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:TemporalSubtraction.as_sql:783",
        "kind": "function",
        "lines": [
          783,
          789
        ],
        "name": "TemporalSubtraction.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:ResolvedOuterRef.as_sql:837",
        "kind": "function",
        "lines": [
          837,
          841
        ],
        "name": "ResolvedOuterRef.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:Func.as_sql:925",
        "kind": "function",
        "lines": [
          925,
          960
        ],
        "name": "Func.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:Value.as_sql:992",
        "kind": "function",
        "lines": [
          992,
          1008
        ],
        "name": "Value.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      },
      {
        "appearance_index": 8,
        "id": "func:django/db/models/expressions.py:RawSQL.as_sql:1059",
        "kind": "function",
        "lines": [
          1059,
          1060
        ],
        "name": "RawSQL.as_sql",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "as_sql"
      }
    ],
    "value_flow_context": [
      {
        "argument_to_parameter": [
          {
            "argument": "condition",
            "parameter": "compiler",
            "position": 0
          }
        ],
        "call": "Case(condition)",
        "call_line": 22,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/aggregates.py:Aggregate.as_sql:103",
          "kind": "function",
          "lines": [
            103,
            128
          ],
          "name": "Aggregate.as_sql",
          "path": "django/db/models/aggregates.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:Case:1319",
          "kind": "class",
          "lines": [
            1377,
            1404
          ],
          "name": "Case",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "connection",
          "template",
          "case_joiner",
          "**extra_context"
        ]
      },
      {
        "argument_to_parameter": [
          {
            "argument": "When(self.expression, then=True)",
            "parameter": "compiler",
            "position": 0
          },
          {
            "argument": "default=False",
            "parameter": "connection",
            "position": 1
          }
        ],
        "call": "Case(When(self.expression, then=True), default=False)",
        "call_line": 8,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/expressions.py:OrderBy.as_oracle:1577",
          "kind": "function",
          "lines": [
            1577,
            1589
          ],
          "name": "OrderBy.as_oracle",
          "path": "django/db/models/expressions.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:Case:1319",
          "kind": "class",
          "lines": [
            1377,
            1404
          ],
          "name": "Case",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "template",
          "case_joiner",
          "**extra_context"
        ]
      },
      {
        "argument_to_parameter": [
          {
            "argument": "When(expr, then=True)",
            "parameter": "compiler",
            "position": 0
          },
          {
            "argument": "default=False",
            "parameter": "connection",
            "position": 1
          }
        ],
        "call": "Case(When(expr, then=True), default=False)",
        "call_line": 8,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/lookups.py:Lookup.as_oracle:137",
          "kind": "function",
          "lines": [
            137,
            148
          ],
          "name": "Lookup.as_oracle",
          "path": "django/db/models/lookups.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:Case:1319",
          "kind": "class",
          "lines": [
            1377,
            1404
          ],
          "name": "Case",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "template",
          "case_joiner",
          "**extra_context"
        ]
      },
      {
        "argument_to_parameter": [
          {
            "argument": "*when_statements",
            "parameter": "compiler",
            "position": 0
          },
          {
            "argument": "output_field=field",
            "parameter": "connection",
            "position": 1
          }
        ],
        "call": "Case(*when_statements, output_field=field)",
        "call_line": 41,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/query.py:QuerySet.bulk_update:855",
          "kind": "function",
          "lines": [
            855,
            905
          ],
          "name": "QuerySet.bulk_update",
          "path": "django/db/models/query.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:Case:1319",
          "kind": "class",
          "lines": [
            1377,
            1404
          ],
          "name": "Case",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "template",
          "case_joiner",
          "**extra_context"
        ]
      }
    ],
    "value_flow_context_policy": "value_flow_context is best-effort implementation evidence from call expressions and signatures; it shows upstream/downstream argument-to-parameter flow but is not a proof of runtime values. Related caller/callee candidates are added to W; read exact nodes before memory_commit/repair."
  },
  "step": 5,
  "visible_thinking": "The issue involves a syntax error in the SQL generated by the `Case` expression when using `~Q(pk__in=[])`. The error message indicates that the SQL generated is malformed, specifically missing the condition in the `WHEN` clause. The `Case` class in `django/db/models/expressions.py` is responsible for generating the SQL for `CASE` expressions. I have already committed the `Case` class to the repair memory. Now I need to examine the `as_sql` method of the `Case` class to understand how it constructs the SQL and identify where the issue might be occurring. The problem seems to be that when `~Q(pk__in=[])` is evaluated, it results in an empty condition, leading to malformed SQL. I should look at how the `When` expressions are processed within the `Case` class and how their SQL is generated."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "confidence": 0.9,
      "evidence_chain": [
        {
          "evidence": "The Case class's as_sql method is responsible for generating the SQL CASE statement. The error occurs when the WHEN clause is empty, suggesting that the method does not properly handle cases where all When conditions compile to empty or invalid SQL fragments.",
          "node_id": "class:django/db/models/expressions.py:Case:1319",
          "role": "target"
        }
      ],
      "failure_seen": "near \"THEN\": syntax error",
      "intent_analysis": "The issue arises when a Case() expression contains When clauses that compile to empty SQL, such as with ~Q(pk__in=[]). The current implementation catches EmptyResultSet to skip such cases, but if a When clause compiles to an empty string instead of raising EmptyResultSet, it gets included in case_parts. When these empty strings are joined, they result in malformed SQL like 'CASE WHEN THEN ...'. The fix is to filter out any empty or whitespace-only case_sql strings from case_parts before joining them, and ensure that if no valid cases remain, the method returns the default SQL directly.",
      "target_nodes": [
        "class:django/db/models/expressions.py:Case:1319"
      ]
    },
    "tool": "repair"
  },
  "result": {
    "cgm_payload": {
      "cgm_insight_count": 0,
      "dispatch_table_count": 0,
      "edge_count": 23,
      "has_adjacency_list": true,
      "has_prompt": true,
      "has_subgraph": true,
      "issue_body_chars": 1385,
      "issue_id": "django__django-15930",
      "issue_title_present": true,
      "node_count": 18,
      "node_paths": [
        "django/db/models/aggregates.py",
        "django/db/models/expressions.py",
        "django/db/models/lookups.py",
        "django/db/models/query.py",
        "django/db/models/sql/where.py"
      ],
      "pending_patch_present": false,
      "plan_target_count": 1,
      "plan_targets": [
        "django/db/models/expressions.py:1319-1409"
      ],
      "plan_text_chars": 1670,
      "planner_confidence": 0.9,
      "prompt_chars": 751,
      "repair_history_count": 0,
      "selected_fix_contract_present": false,
      "serialized_code_count": 1,
      "snippet_context_count": 0,
      "snippet_count": 1,
      "snippet_target_count": 1
    },
    "cgm_response": {
      "model": "qwen3-235b-a22b-thinking-2507",
      "output_format": "json_patch",
      "raw_preview": "{\n  \"summary\": \"Skip empty WHEN conditions in Case.as_sql to avoid invalid SQL\",\n  \"edits\": [\n    {\n      \"path\": \"django/db/models/expressions.py\",\n      \"start\": 1386,\n      \"end\": 1392,\n      \"new_text\": \"        for case in self.cases:\\\\n            try:\\\\n                case_sql, case_params = compiler.compile(case)\\\\n            except EmptyResultSet:\\\\n                continue\\\\n            # Skip if the condition is empty (which would cause invalid SQL)\\\\n            if not case_sql.strip():\\\\n                continue\\\\n            case_parts.append(case_sql)\\\\n            sql_params.extend(case_params)\"\n    }\n  ]\n}",
      "reasoning_chars": 35796,
      "reasoning_preview": "We are given an issue: Case() crashes with ~Q(pk__in=[]). The problem is that when we have a When condition that compiles to an empty string (like ~Q(pk__in=[])), the generated SQL becomes \"CASE WHEN THEN ...\", which is invalid.\n\nLooking at the provided code snippet for `Case.as_sql` (lines 1377-1404):\n\n```python\n1377:     def as_sql(\n1378:         self, compiler, connection, template=None, case_joiner=None, **extra_context\n1379:     ):\n1380:         connection.ops.check_expression_support(self)\n1381:         if not self.cases:\n1382:             return compiler.compile(self.default)\n1383:         template_params = {**self.extra, **extra_context}\n1384:         case_parts = []\n1385:         sql_params = []\n1386:         for case in self.cases:\n1387:             try:\n1388:                 case_sql, case_params = compiler.compile(case)\n1389:             except EmptyResultSet:\n1390:                 continue\n1391:             case_parts.append(case_sql)\n1392:             sql_params.extend(case_params)\n1393:         default_sql, default_params = compiler.compile(self.default)\n1394:         if not case_parts:\n1395:             return default_sql, default_params\n1396:         case_joiner =",
      "thinking_enabled": true
    },
    "error_origin": "generated_patch_behavior",
    "failure_feedback": {
      "error_summary": "Skip empty WHEN conditions in Case.as_sql to avoid invalid SQL\nexception_types: sqlite3.OperationalError; django.db.utils.OperationalError\nactual_messages: near \"THEN\": syntax error\nActual runtime failure summary (benchmark expected values omitted):\nexception_types: sqlite3.OperationalError, django.db.utils.OperationalError\nactual_messages:\n- near \"THEN\": syntax error",
      "failed_patch": {
        "edit_count": 1,
        "edits": [
          {
            "end": 1392,
            "new_text": "for case in self.cases:\n            try:\n                case_sql, case_params = compiler.compile(case)\n            except EmptyResultSet:\n                continue\n            # Skip if the condition is empty (which would cause invalid SQL)\n            if not case_sql.strip():\n                continue\n            case_parts.append(case_sql)\n            sql_params.extend(case_params)",
            "path": "django/db/models/expressions.py",
            "start": 1386
          }
        ],
        "summary": "Skip empty WHEN conditions in Case.as_sql to avoid invalid SQL"
      },
      "failed_tests": []
    },
    "patch_preview": {
      "edit_count": 1,
      "edits": [
        {
          "end": 1392,
          "new_text": "for case in self.cases:\n            try:\n                case_sql, case_params = compiler.compile(case)\n            except EmptyResultSet:\n                continue\n            # Skip if the condition is empty (which would cause invalid SQL)\n            if not case_sql.strip():\n                continue\n            case_parts.append(case_sql)\n            sql_params.extend(case_params)",
          "path": "django/db/models/expressions.py",
          "start": 1386
        }
      ],
      "summary": "Skip empty WHEN conditions in Case.as_sql to avoid invalid SQL"
    },
    "rolled_back": true,
    "source_tree_state": "rolled_back_to_original",
    "status": "test_failed",
    "summary": "Skip empty WHEN conditions in Case.as_sql to avoid invalid SQL",
    "test_summary": {
      "command": "<official SWE-bench eval command omitted; contains benchmark harness setup/test patch>",
      "command_omitted_for_benchmark_hygiene": true,
      "excerpt": "Actual runtime failure summary (benchmark expected values omitted):\nexception_types: sqlite3.OperationalError, django.db.utils.OperationalError\nactual_messages:\n- near \"THEN\": syntax error",
      "implementation_frames": [],
      "parser_error": "missing version for SWE-bench log parser",
      "resolved": null,
      "returncode": 1,
      "runtime_observations": {
        "actual_assertion_values": [],
        "actual_messages": [
          "near \"THEN\": syntax error"
        ],
        "exception_types": [
          "sqlite3.OperationalError",
          "django.db.utils.OperationalError"
        ],
        "failed_selectors": [],
        "omitted_hidden_expected_values": false,
        "policy": "actual runtime output only; benchmark test source and hidden expected values are omitted"
      },
      "status": "failed",
      "tests_status": {}
    },
    "tool": "repair",
    "touched_paths": [
      "django/db/models/expressions.py"
    ]
  },
  "step": 6,
  "visible_thinking": "The issue involves a `Case()` expression crashing when used with `~Q(pk__in=[])`, resulting in malformed SQL with a syntax error near \"THEN\". The root cause appears to be in the `as_sql` method of the `Case` class, where the SQL generation logic fails to properly handle empty conditions.\n\nFrom the code read, the `Case.as_sql` method iterates through `self.cases` and compiles each case. If a case raises `EmptyResultSet`, it's skipped. However, if all cases are skipped, the method still attempts to join `case_parts`, which could result in an empty string or improper SQL structure, leading to the syntax error.\n\nThe error message indicates:\n```\nProgrammingError: syntax error at or near \"THEN\"\nLINE 1: ..._user\".\"id\" FROM \"users_user\" ORDER BY CASE WHEN THEN true ...\n```\nThis suggests that the `WHEN` clause is empty, producing `CASE WHEN THEN ...`.\n\nLooking at the `as_sql` method:\n- It processes `self.cases`.\n- If a case raises `EmptyResultSet`, it's skipped (`continue`).\n- If all cases are skipped, `case_parts` remains empty.\n- Later, `case_joiner.join(case_parts)` would produce an empty string.\n- The template likely expects a non-empty condition, causing the malformed SQL.\n\nThe fix should ensure that when all cases are skipped (i.e., `case_parts` is empty), the method returns the default value's SQL directly, similar to how it handles `if not self.cases:` at the beginning. Currently, there's a check `if not case_parts:` after processing, but it only returns `default_sql` if `case_parts` is empty *after* the loop. This seems correct, but perhaps the issue lies elsewhere, such as in how individual `When` clauses are compiled when their condition is an empty `~Q(pk__in=[])`.\n\nLet's examine the `When` class and its `as_sql` method to see how it handles conditions that might raise `EmptyResultSet`. Since the `Case` class catches `EmptyResultSet` from `compiler.compile(case)`, it skips those cases. If all `When` conditions are empty, `case_parts` becomes empty, and the subsequent logic should return just the default. But the error suggests that an empty `WHEN` clause is being included in the SQL.\n\nPerhaps the problem is that the `When` clause itself, when compiled, produces a malformed SQL fragment (like `WHEN THEN`) instead of raising `EmptyResultSet`. This would mean the `Case` class doesn't skip it, and the malformed fragment gets included.\n\nTo confirm, I should look at the `When` class's `as_sql` method. However, since the focus is on the `Case` class and the error originates there, I'll proceed with the repair intent focused on ensuring that if `case_parts` results in an empty list after filtering, the method correctly returns only the default SQL, or handles the template substitution to avoid malformed SQL.\n\nBut looking closely at the code:\n```python\nif not case_parts:\n    return default_sql, default_params\n```\nThis check exists and should handle the case where all `When` clauses are skipped. So, the issue might be that the `When` clause isn't raising `EmptyResultSet` but instead producing invalid SQL.\n\nGiven the task is to fix the `Case` class, and assuming the `When` class behavior is as-is, the `Case.as_sql` method should robustly handle cases where the compiled SQL for a `When` clause is effectively empty or invalid, beyond just catching `EmptyResultSet`.\n\nHowever, the most straightforward fix, based on the current logic and the error, is to ensure that if `case_parts` leads to an empty string when joined, the method doesn't proceed to template substitution that would create `CASE WHEN THEN ...`. The existing check `if not case_parts:` should already cover this, so perhaps the issue is in the template or how `template_params[\"cases\"]` is used.\n\nLooking at the code:\n```python\ntemplate_params[\"cases\"] = case_joiner.join(case_parts)\n...\nsql = template % template_params\n```\nIf `case_parts` is empty, `cases` becomes an empty string. The template likely has something like `CASE WHEN %(cases)s THEN ...`, and substituting an empty string for `%(cases)s` would result in `CASE WHEN THEN ...`.\n\nThus, the fix should prevent the template from being used if `case_parts` is empty after processing, or modify the template usage to handle empty cases gracefully. Since the check `if not case_parts:` exists, it should work, but maybe it's bypassed or the logic flows differently.\n\nWait, let's re-examine the code:\n```python\n# Line 1381: if not self.cases:\n#     return compiler.compile(self.default)\n# ...\n# Line 1394: if not case_parts:\n#     return default_sql, default_params\n```\nBoth checks seem to handle returning the default. If `self.cases` is not empty initially, but all cases raise `EmptyResultSet` (and thus are skipped), `case_parts` becomes empty, and the second check should return the default.\n\nBut the error occurs, meaning the flow doesn't reach that point correctly, or the template substitution happens regardless. Perhaps the issue is that even though `case_parts` is empty, the code proceeds to template substitution because the check is after the template params are set up.\n\nActually, looking at lines:\n```python\ncase_joiner = case_joiner or self.case_joiner\ntemplate_params[\"cases\"] = case_joiner.join(case_parts)\ntemplate_params[\"default\"] = default_sql\nsql_params.extend(default_params)\ntemplate = template or template_params.get(\"template\", self.template)\nsql = template % template_params\n```\nIf `case_parts` is empty, `template_params[\"cases\"]` becomes an empty string. When `template % template_params` is executed, if the template is something like `\"CASE WHEN %(cases)s THEN ... END\"`, substituting an empty string for `%(cases)s` gives `\"CASE WHEN THEN ... END\"`.\n\nSo, the fix should ensure that if `case_parts` is empty, we don't proceed with the template that expects a non-empty `cases` part. The existing check `if not case_parts: return default_sql, default_params` should handle this, but it's placed before setting `template_params[\"cases\"]` and proceeding with template substitution.\n\nWait, no:\n```python\n# Line 1394: if not case_parts:\n#     return default_sql, default_params\n```\nThis check is after `template_params[\"cases\"]` is set. So if `case_parts` is empty, it returns `default_sql` immediately, skipping the template substitution. That should work.\n\nUnless there's a bug in the placement or the condition is not met. Let me double-check the line numbers in the code snippet:\nLines 1394-1395:\n```python\nif not case_parts:\n    return default_sql, default_params\n```\nAnd lines 1397-1401:\n```python\ncase_joiner = case_joiner or self.case_joiner\ntemplate_params[\"cases\"] = case_joiner.join(case_parts)\ntemplate_params[\"default\"] = default_sql\nsql_params.extend(default_params)\ntemplate = template or template_params.get(\"template\", self.template)\nsql = template % template_params\n```\nSo, the check is indeed before the template substitution. Therefore, if `case_parts` is empty, it should return `default_sql`. But the error suggests it doesn't.\n\nPerhaps the issue is that `case_parts` is not empty, but contains an empty string or a malformed fragment. For example, if a `When` clause compiles to an empty string instead of raising `EmptyResultSet`, it gets appended to `case_parts`, and `case_joiner.join(case_parts)` produces an empty string, leading to the malformed SQL.\n\nThus, the fix should also check if any element in `case_parts` is empty or invalid and handle it appropriately.\n\nAlternatively, the `When` class's `as_sql` method might not be raising `EmptyResultSet` for `~Q(pk__in=[])` as expected, and instead returns an empty or malformed SQL string.\n\nSince I can't modify the `When` class directly without evidence, I'll focus on making the `Case.as_sql` method more robust by ensuring that `case_parts` doesn't contain empty strings before proceeding.\n\nThe repair will involve modifying the `Case.as_sql` method to filter out any empty or whitespace-only `case_sql` strings from `case_parts` before joining them, and also ensure that if after filtering, `case_parts` is empty, it returns the default.\n\nThis approach addresses the symptom (malformed SQL due to empty `WHEN` clause) by preventing empty `case_sql` from being included in the final SQL construction."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "find_type": "class",
      "path_glob": "**/*.py",
      "query": "When"
    },
    "tool": "explore_find"
  },
  "result": {
    "path_glob": "**/*.py",
    "result_policy": "function/class/method/assignment results include a small implementation preview in W for orientation only; read the node before memory_commit or repair evidence. file-level results list top symbols instead of full file text",
    "results": [
      {
        "code": "1245: class When(Expression):\n1246:     template = \"WHEN %(condition)s THEN %(result)s\"\n1247:     # This isn't a complete conditional expression, must be used in Case().\n1248:     conditional = False\n1249: \n1250:     def __init__(self, condition=None, then=None, **lookups):\n1251:         if lookups:\n1252:             if condition is None:\n1253:                 condition, lookups = Q(**lookups), None\n1254:             elif getattr(condition, \"conditional\", False):\n1255:                 condition, lookups = Q(condition, **lookups), None\n1256:         if condition is None or not getattr(condition, \"conditional\", False) or lookups:\n1257:             raise TypeError(\n1258:                 \"When() supports a Q object, a boolean expression, or lookups \"\n1259:                 \"as a condition.\"\n1260:             )\n1261:         if isinstance(condition, Q) and not condition:\n1262:             raise ValueError(\"An empty Q() can't be used as a When() condition.\")\n1263:         super().__init__(output_field=None)\n1264:         self.condition = condition\n1265:         self.result = self._parse_expressions(then)[0]\n1266: \n1267:     def __str__(self):\n1268:         return \"WHEN %r THEN %r\" % (self.condition, self.result)\n1269: \n1270:     def __repr__(self):\n1271:         return \"<%s: %s>\" % (self.__class__.__name__, self)\n1272: ",
        "code_preview_lines": [
          1245,
          1272
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/db/models/expressions.py:When:1245",
        "kind": "class",
        "lines": [
          1245,
          1315
        ],
        "local_symbol_references": [
          {
            "appearance_index": 11,
            "id": "class:django/db/models/expressions.py:Expression:462",
            "kind": "class",
            "lines": [
              462,
              490
            ],
            "name": "Expression",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "Expression"
          },
          {
            "appearance_index": 104,
            "id": "func:django/db/models/expressions.py:BaseExpression.conditional:285",
            "kind": "function",
            "lines": [
              285,
              286
            ],
            "name": "BaseExpression.conditional",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "conditional"
          },
          {
            "appearance_index": 144,
            "id": "class:django/db/models/expressions.py:Case:1319",
            "kind": "class",
            "lines": [
              1319,
              1409
            ],
            "name": "Case",
            "path": "django/db/models/expressions.py",
            "read_status": "read",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "Case"
          },
          {
            "appearance_index": 883,
            "id": "func:django/db/models/expressions.py:BaseExpression.output_field:293",
            "kind": "function",
            "lines": [
              293,
              299
            ],
            "name": "BaseExpression.output_field",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "output_field"
          },
          {
            "appearance_index": 964,
            "id": "func:django/db/models/expressions.py:BaseExpression._parse_expressions:200",
            "kind": "function",
            "lines": [
              200,
              206
            ],
            "name": "BaseExpression._parse_expressions",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_parse_expressions"
          }
        ],
        "name": "When",
        "path": "django/db/models/expressions.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 11,
            "id": "class:django/db/models/expressions.py:Expression:462",
            "kind": "class",
            "lines": [
              462,
              490
            ],
            "name": "Expression",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "Expression"
          },
          {
            "appearance_index": 104,
            "id": "func:django/db/models/expressions.py:BaseExpression.conditional:285",
            "kind": "function",
            "lines": [
              285,
              286
            ],
            "name": "BaseExpression.conditional",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "conditional"
          },
          {
            "appearance_index": 883,
            "id": "func:django/db/models/expressions.py:BaseExpression.output_field:293",
            "kind": "function",
            "lines": [
              293,
              299
            ],
            "name": "BaseExpression.output_field",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "output_field"
          },
          {
            "appearance_index": 964,
            "id": "func:django/db/models/expressions.py:BaseExpression._parse_expressions:200",
            "kind": "function",
            "lines": [
              200,
              206
            ],
            "name": "BaseExpression._parse_expressions",
            "path": "django/db/models/expressions.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_parse_expressions"
          }
        ]
      }
    ],
    "tool": "explore_find",
    "warning": null
  },
  "step": 7,
  "visible_thinking": "The previous repair attempt failed because the patch didn't correctly handle the case where a `When` clause compiles to an empty string, leading to invalid SQL. The error message \"near 'THEN': syntax error\" indicates that the SQL generated was malformed, specifically missing the condition in the `WHEN` clause.\nThe failed patch tried to skip empty `case_sql` by checking `if not case_sql.strip(): continue`, but it seems this wasn't sufficient or placed correctly.\nLet's analyze the `Case.as_sql` method more carefully. The issue is in the loop where `case_sql` is compiled and appended to `case_parts`. If `case_sql` is empty, it shouldn't be appended. However, the check might not be comprehensive enough or placed at the right point.\nI need to look at the `When` class and how it compiles to see if there's a better way to detect and handle empty conditions. Since the previous patch was applied and rolled back, I should focus on understanding why it didn't work and then propose a revised approach.\nThe key is to ensure that any `When` clause that results in an empty SQL string is skipped entirely, not just caught by `EmptyResultSet`.\nLet's consider exploring the `When` class to understand its `as_sql` method, as this is where the empty SQL might originate."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "node_id": "class:django/db/models/expressions.py:When:1245",
      "view": "body"
    },
    "tool": "read"
  },
  "result": {
    "code": "1245: class When(Expression):\n1246:     template = \"WHEN %(condition)s THEN %(result)s\"\n1247:     # This isn't a complete conditional expression, must be used in Case().\n1248:     conditional = False\n1249: \n1250:     def __init__(self, condition=None, then=None, **lookups):\n1251:         if lookups:\n1252:             if condition is None:\n1253:                 condition, lookups = Q(**lookups), None\n1254:             elif getattr(condition, \"conditional\", False):\n1255:                 condition, lookups = Q(condition, **lookups), None\n1256:         if condition is None or not getattr(condition, \"conditional\", False) or lookups:\n1257:             raise TypeError(\n1258:                 \"When() supports a Q object, a boolean expression, or lookups \"\n1259:                 \"as a condition.\"\n1260:             )\n1261:         if isinstance(condition, Q) and not condition:\n1262:             raise ValueError(\"An empty Q() can't be used as a When() condition.\")\n1263:         super().__init__(output_field=None)\n1264:         self.condition = condition\n1265:         self.result = self._parse_expressions(then)[0]\n1266: \n1267:     def __str__(self):\n1268:         return \"WHEN %r THEN %r\" % (self.condition, self.result)\n1269: \n1270:     def __repr__(self):\n1271:         return \"<%s: %s>\" % (self.__class__.__name__, self)\n1272: \n1273:     def get_source_expressions(self):\n1274:         return [self.condition, self.result]\n1275: \n1276:     def set_source_expressions(self, exprs):\n1277:         self.condition, self.result = exprs\n1278: \n1279:     def get_source_fields(self):\n1280:         # We're only interested in the fields of the result expressions.\n1281:         return [self.result._output_field_or_none]\n1282: \n1283:     def resolve_expression(\n1284:         self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False\n1285:     ):\n1286:         c = self.copy()\n1287:         c.is_summary = summarize\n1288:         if hasattr(c.condition, \"resolve_expression\"):\n1289:             c.condition = c.condition.resolve_expression(\n1290:                 query, allow_joins, reuse, summarize, False\n1291:             )\n1292:         c.result = c.result.resolve_expression(\n1293:             query, allow_joins, reuse, summarize, for_save\n1294:         )\n1295:         return c\n1296: \n1297:     def as_sql(self, compiler, connection, template=None, **extra_context):\n1298:         connection.ops.check_expression_support(self)\n1299:         template_params = extra_context\n1300:         sql_params = []\n1301:         condition_sql, condition_params = compiler.compile(self.condition)\n1302:         template_params[\"condition\"] = condition_sql\n1303:         sql_params.extend(condition_params)\n1304:         result_sql, result_params = compiler.compile(self.result)\n1305:         template_params[\"result\"] = result_sql\n1306:         sql_params.extend(result_params)\n1307:         template = template or self.template\n1308:         return template % template_params, sql_params\n1309: \n1310:     def get_group_by_cols(self, alias=None):\n1311:         # This is not a complete expression and cannot be used in GROUP BY.\n1312:         cols = []\n1313:         for source in self.get_source_expressions():\n1314:             cols.extend(source.get_group_by_cols())\n1315:         return cols",
    "dispatch_relationship_context": [],
    "dispatch_tables": [],
    "local_symbol_references": [
      {
        "appearance_index": 11,
        "id": "class:django/db/models/expressions.py:Expression:462",
        "kind": "class",
        "lines": [
          462,
          490
        ],
        "name": "Expression",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "Expression"
      },
      {
        "appearance_index": 104,
        "id": "func:django/db/models/expressions.py:BaseExpression.conditional:285",
        "kind": "function",
        "lines": [
          285,
          286
        ],
        "name": "BaseExpression.conditional",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "conditional"
      },
      {
        "appearance_index": 144,
        "id": "class:django/db/models/expressions.py:Case:1319",
        "kind": "class",
        "lines": [
          1319,
          1409
        ],
        "name": "Case",
        "path": "django/db/models/expressions.py",
        "read_status": "read",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "Case"
      },
      {
        "appearance_index": 883,
        "id": "func:django/db/models/expressions.py:BaseExpression.output_field:293",
        "kind": "function",
        "lines": [
          293,
          299
        ],
        "name": "BaseExpression.output_field",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "output_field"
      },
      {
        "appearance_index": 964,
        "id": "func:django/db/models/expressions.py:BaseExpression._parse_expressions:200",
        "kind": "function",
        "lines": [
          200,
          206
        ],
        "name": "BaseExpression._parse_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "_parse_expressions"
      },
      {
        "appearance_index": 1175,
        "id": "func:django/db/models/expressions.py:BaseExpression.get_source_expressions:194",
        "kind": "function",
        "lines": [
          194,
          195
        ],
        "name": "BaseExpression.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      },
      {
        "appearance_index": 1175,
        "id": "func:django/db/models/expressions.py:CombinedExpression.get_source_expressions:641",
        "kind": "function",
        "lines": [
          641,
          642
        ],
        "name": "CombinedExpression.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      },
      {
        "appearance_index": 1175,
        "id": "func:django/db/models/expressions.py:Func.get_source_expressions:908",
        "kind": "function",
        "lines": [
          908,
          909
        ],
        "name": "Func.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      }
    ],
    "node": {
      "id": "class:django/db/models/expressions.py:When:1245",
      "kind": "class",
      "lines": [
        1245,
        1315
      ],
      "name": "When",
      "path": "django/db/models/expressions.py"
    },
    "relationship_context_policy": "consumer candidates are auto-added to W as orientation code only; read the exact node before memory_commit/repair evidence",
    "tool": "read",
    "unread_local_symbol_references": [
      {
        "appearance_index": 11,
        "id": "class:django/db/models/expressions.py:Expression:462",
        "kind": "class",
        "lines": [
          462,
          490
        ],
        "name": "Expression",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "Expression"
      },
      {
        "appearance_index": 104,
        "id": "func:django/db/models/expressions.py:BaseExpression.conditional:285",
        "kind": "function",
        "lines": [
          285,
          286
        ],
        "name": "BaseExpression.conditional",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "conditional"
      },
      {
        "appearance_index": 883,
        "id": "func:django/db/models/expressions.py:BaseExpression.output_field:293",
        "kind": "function",
        "lines": [
          293,
          299
        ],
        "name": "BaseExpression.output_field",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "output_field"
      },
      {
        "appearance_index": 964,
        "id": "func:django/db/models/expressions.py:BaseExpression._parse_expressions:200",
        "kind": "function",
        "lines": [
          200,
          206
        ],
        "name": "BaseExpression._parse_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "_parse_expressions"
      },
      {
        "appearance_index": 1175,
        "id": "func:django/db/models/expressions.py:BaseExpression.get_source_expressions:194",
        "kind": "function",
        "lines": [
          194,
          195
        ],
        "name": "BaseExpression.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      },
      {
        "appearance_index": 1175,
        "id": "func:django/db/models/expressions.py:CombinedExpression.get_source_expressions:641",
        "kind": "function",
        "lines": [
          641,
          642
        ],
        "name": "CombinedExpression.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      },
      {
        "appearance_index": 1175,
        "id": "func:django/db/models/expressions.py:Func.get_source_expressions:908",
        "kind": "function",
        "lines": [
          908,
          909
        ],
        "name": "Func.get_source_expressions",
        "path": "django/db/models/expressions.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_source_expressions"
      }
    ],
    "value_flow_context": [
      {
        "argument_to_parameter": [
          {
            "argument": "self.filter",
            "parameter": "condition",
            "position": 0
          },
          {
            "argument": "source_expressions[0]",
            "parameter": "then",
            "position": 1
          }
        ],
        "call": "When(self.filter, then=source_expressions[0])",
        "call_line": 21,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/aggregates.py:Aggregate.as_sql:103",
          "kind": "function",
          "lines": [
            103,
            128
          ],
          "name": "Aggregate.as_sql",
          "path": "django/db/models/aggregates.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:When:1245",
          "kind": "class",
          "lines": [
            1245,
            1315
          ],
          "name": "When",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "**lookups"
        ]
      },
      {
        "argument_to_parameter": [
          {
            "argument": "self.expression",
            "parameter": "condition",
            "position": 0
          },
          {
            "argument": "True",
            "parameter": "then",
            "position": 1
          }
        ],
        "call": "When(self.expression, then=True)",
        "call_line": 9,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/expressions.py:OrderBy.as_oracle:1577",
          "kind": "function",
          "lines": [
            1577,
            1589
          ],
          "name": "OrderBy.as_oracle",
          "path": "django/db/models/expressions.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:When:1245",
          "kind": "class",
          "lines": [
            1245,
            1315
          ],
          "name": "When",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "**lookups"
        ]
      },
      {
        "argument_to_parameter": [
          {
            "argument": "expr",
            "parameter": "condition",
            "position": 0
          },
          {
            "argument": "True",
            "parameter": "then",
            "position": 1
          }
        ],
        "call": "When(expr, then=True)",
        "call_line": 8,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/lookups.py:Lookup.as_oracle:137",
          "kind": "function",
          "lines": [
            137,
            148
          ],
          "name": "Lookup.as_oracle",
          "path": "django/db/models/lookups.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:When:1245",
          "kind": "class",
          "lines": [
            1245,
            1315
          ],
          "name": "When",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "**lookups"
        ]
      },
      {
        "argument_to_parameter": [
          {
            "argument": "pk=obj.pk",
            "parameter": "condition",
            "position": 0
          },
          {
            "argument": "attr",
            "parameter": "then",
            "position": 1
          }
        ],
        "call": "When(pk=obj.pk, then=attr)",
        "call_line": 40,
        "relation": "value_flow_upstream",
        "source": {
          "id": "func:django/db/models/query.py:QuerySet.bulk_update:855",
          "kind": "function",
          "lines": [
            855,
            905
          ],
          "name": "QuerySet.bulk_update",
          "path": "django/db/models/query.py"
        },
        "source_read_status": "unread",
        "target": {
          "id": "class:django/db/models/expressions.py:When:1245",
          "kind": "class",
          "lines": [
            1245,
            1315
          ],
          "name": "When",
          "path": "django/db/models/expressions.py"
        },
        "target_read_status": "read",
        "unmapped_parameters": [
          "**lookups"
        ]
      }
    ],
    "value_flow_context_policy": "value_flow_context is best-effort implementation evidence from call expressions and signatures; it shows upstream/downstream argument-to-parameter flow but is not a proof of runtime values. Related caller/callee candidates are added to W; read exact nodes before memory_commit/repair."
  },
  "step": 8,
  "visible_thinking": "The previous repair attempt failed because the patch did not correctly handle the case where a `When` clause compiles to an empty string. The error message indicates a syntax error near \"THEN\", which suggests that the generated SQL has an incomplete `WHEN` clause.\nI need to examine the `When` class more closely to understand how it generates its SQL, particularly in the scenario where the condition is an empty `Q` object or evaluates to an empty string. The issue seems to stem from the `When` class's `__init__` method, which raises a `ValueError` for empty `Q` objects, but the problem arises with `~Q(pk__in=[])`.\nLet's read the `When` class implementation to see how it handles conditions and generates SQL."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "note": "The When class's as_sql method is where the empty condition SQL originates, contributing to the malformed CASE statement in Case.as_sql.",
      "select_ids": [
        "class:django/db/models/expressions.py:When:1245"
      ]
    },
    "tool": "memory_commit"
  },
  "result": {
    "already_present_ids": [],
    "committed": [
      "class:django/db/models/expressions.py:When:1245"
    ],
    "dropped_by_keep_ids": [],
    "explicitly_selected_ids": [
      "class:django/db/models/expressions.py:When:1245"
    ],
    "memory": [
      {
        "has_code": true,
        "id": "class:django/db/models/expressions.py:Case:1319",
        "kind": "class",
        "lines": [
          1319,
          1409
        ],
        "name": "Case",
        "path": "django/db/models/expressions.py"
      },
      {
        "has_code": true,
        "id": "class:django/db/models/expressions.py:When:1245",
        "kind": "class",
        "lines": [
          1245,
          1315
        ],
        "name": "When",
        "path": "django/db/models/expressions.py"
      }
    ],
    "memory_changed": true,
    "newly_added_ids": [
      "class:django/db/models/expressions.py:When:1245"
    ],
    "tool": "memory_commit"
  },
  "step": 9,
  "visible_thinking": "The previous repair attempt failed because the patch did not correctly handle the case where a `When` clause compiles to an empty string, leading to invalid SQL. The error message \"near 'THEN': syntax error\" indicates that the generated SQL had a structure like `CASE WHEN THEN ...`, which is invalid.\nThe current repair memory (`M`) contains the `Case` class, which is the target of the repair. The `When` class has been read but not yet committed to `M`. The issue seems to stem from how `When` objects are compiled within the `Case.as_sql` method.\nThe `Case.as_sql` method iterates through `self.cases`, compiles each one, and catches `EmptyResultSet` to skip them. However, it doesn't check if the compiled SQL is empty or whitespace-only before adding it to `case_parts`. This is the root cause of the malformed SQL.\nThe previous patch tried to add a check `if not case_sql.strip(): continue` but it seems that didn't fully address the issue or introduced a new problem.\nLet's re-examine the `When` class to understand its `as_sql` method, as this is where the empty SQL might originate. The `When` class's `as_sql` method uses a template `\"WHEN %(condition)s THEN %(result)s\"`. If `condition_sql` (from `compiler.compile(self.condition)`) is empty, this would result in `WHEN THEN ...`.\nThe problem is likely that `compiler.compile(self.condition)` for `~Q(pk__in=[])` returns an empty string for `condition_sql` instead of raising `EmptyResultSet`. Therefore, the check should be in the `When.as_sql` method or the `Case.as_sql` method should be more robust.\nSince the `Case` class is already in `M` and is the direct location of the SQL assembly, it's the most appropriate place to fix this. The previous attempt modified the loop in `Case.as_sql`. Let's refine that approach.\nThe key insight is that if `condition_sql` is empty, the entire `When` clause should be skipped, not just appended as an empty string. The check should be more specific to detect when the `condition` part of a `When` is effectively empty.\nI should commit the `When` class to `M` as well, since understanding its `as_sql` method is crucial to fixing the issue. Then, I can propose a repair that ensures `Case.as_sql` properly handles `When` objects that compile to empty conditions.\nThe `When` class's `as_sql` method (lines 1297-1308) gets `condition_sql` from `compiler.compile(self.condition)`. If this is empty, the template `\"WHEN %(condition)s THEN %(result)s\"` will produce `\"WHEN THEN ...\"`.\nSo, the fix could be in `When.as_sql` to raise `EmptyResultSet` if `condition_sql` is empty, or in `Case.as_sql` to check the compiled `case_sql` for emptiness before appending.\nGiven that `When` is a general-purpose class, modifying `Case.as_sql` to be more defensive seems safer. The previous patch's logic was on the right track but perhaps the check wasn't sufficient or placed correctly.\nLet's commit the `When` class to `M` to make it part of the evidence chain, then revise the repair intent to be more precise about checking for empty conditions in `Case.as_sql`.\nThe goal is to prevent `case_parts` from containing entries that would result in `WHEN THEN` in the final SQL. This means checking if the compiled `case_sql` is effectively empty (e.g., just whitespace or the template with empty substitutions) and skipping it.\nThe template for `When` is `\"WHEN %(condition)s THEN %(result)s\"`. If `condition` is empty, it becomes `\"WHEN THEN ...\"`. So, a good check in `Case.as_sql` would be to see if `case_sql` starts with `\"WHEN THEN\"` or is just whitespace.\nHowever, a more robust check would be to ensure that the `condition` part of the `When` is not empty. Since we have access to the `When` object in the loop, we could potentially inspect `case.condition` before compiling, but that might be complex.\nA simpler and more direct fix is to enhance the check in `Case.as_sql` to ensure that `case_sql` is not just whitespace and actually contains a non-trivial condition.\nLet's proceed with committing the `When` class and then proposing a refined repair."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "confidence": 0.95,
      "evidence_chain": [
        {
          "evidence": "The When class's as_sql method generates the SQL for individual WHEN clauses. When the condition compiles to an empty string (e.g., from ~Q(pk__in=[])), it produces invalid SQL like 'WHEN THEN ...'. The method should detect this and raise EmptyResultSet to signal that this clause should be skipped by the Case class.",
          "node_id": "class:django/db/models/expressions.py:When:1245",
          "role": "target"
        },
        {
          "evidence": "The Case class's as_sql method iterates over self.cases and compiles each one, catching EmptyResultSet to skip invalid cases. However, if a When clause returns an empty condition SQL without raising EmptyResultSet, it leads to malformed CASE statements.",
          "node_id": "class:django/db/models/expressions.py:Case:1319",
          "role": "entry"
        }
      ],
      "failure_seen": "near \"THEN\": syntax error",
      "intent_analysis": "Modify the When.as_sql method to check if the compiled condition SQL is empty. If it is, raise EmptyResultSet so that the Case class can properly skip this clause, preventing the generation of invalid SQL like 'CASE WHEN THEN ...'.",
      "target_nodes": [
        "class:django/db/models/expressions.py:When:1245"
      ]
    },
    "tool": "repair_propose"
  },
  "result": {
    "cgm_payload": {
      "cgm_insight_count": 1,
      "dispatch_table_count": 0,
      "edge_count": 47,
      "has_adjacency_list": true,
      "has_prompt": true,
      "has_subgraph": true,
      "issue_body_chars": 1385,
      "issue_id": "django__django-15930",
      "issue_title_present": true,
      "node_count": 30,
      "node_paths": [
        "django/db/models/aggregates.py",
        "django/db/models/expressions.py",
        "django/db/models/lookups.py",
        "django/db/models/query.py",
        "django/db/models/sql/where.py"
      ],
      "pending_patch_present": false,
      "plan_target_count": 1,
      "plan_targets": [
        "django/db/models/expressions.py:1245-1315"
      ],
      "plan_text_chars": 7978,
      "planner_confidence": 0.95,
      "prompt_chars": 751,
      "repair_history_count": 1,
      "selected_fix_contract_present": false,
      "serialized_code_count": 2,
      "snippet_context_count": 1,
      "snippet_count": 2,
      "snippet_target_count": 1
    },
    "cgm_response": {
      "model": "qwen3-235b-a22b-thinking-2507",
      "output_format": "json_patch",
      "raw_preview": "{\n  \"summary\": \"Replace empty condition SQL in When.as_sql with '1=1' to avoid invalid CASE syntax\",\n  \"edits\": [\n    {\n      \"path\": \"django/db/models/expressions.py\",\n      \"start\": 1301,\n      \"end\": 1303,\n      \"new_text\": \"        condition_sql, condition_params = compiler.compile(self.condition)\\n        if not condition_sql.strip():\\n            condition_sql = '1=1'\\n            condition_params = []\\n        template_params[\\\"condition\\\"] = condition_sql\\n        sql_params.extend(condition_params)\"\n    }\n  ]\n}",
      "reasoning_chars": 31020,
      "reasoning_preview": "We are given an issue: Case() crashes when using ~Q(pk__in=[]). The problem is that the condition in the When clause becomes empty, leading to SQL like \"CASE WHEN THEN ...\", which is invalid.\n\nLooking at the provided code:\n\nIn `When.as_sql` (lines 1297-1309):\n```python\n1297:     def as_sql(self, compiler, connection, template=None, **extra_context):\n1298:         connection.ops.check_expression_support(self)\n1299:         template_params = extra_context\n1300:         sql_params = []\n1301:         condition_sql, condition_params = compiler.compile(self.condition)\n1302:         template_params[\"condition\"] = condition_sql\n1303:         sql_params.extend(condition_params)\n1304:         result_sql, result_params = compiler.compile(self.result)\n1305:         template_params[\"result\"] = result_sql\n1306:         sql_params.extend(result_params)\n1307:         template = template or self.template\n1308:         return template % template_params, sql_params\n```\n\nThe issue is that when `self.condition` is `~Q(pk__in=[])`, the compiled `condition_sql` becomes an empty string. Then, the template `\"WHEN %(condition)s THEN %(result)s\"` becomes `\"WHEN  THEN ...\"` which is invalid.\n\nThe planner guid",
      "thinking_enabled": true
    },
    "done": false,
    "patch_preview": {
      "edit_count": 1,
      "edits": [
        {
          "end": 1303,
          "new_text": "condition_sql, condition_params = compiler.compile(self.condition)\n        if not condition_sql.strip():\n            condition_sql = '1=1'\n            condition_params = []\n        template_params[\"condition\"] = condition_sql\n        sql_params.extend(condition_params)",
          "path": "django/db/models/expressions.py",
          "start": 1301
        }
      ],
      "summary": "Replace empty condition SQL in When.as_sql with '1=1' to avoid invalid CASE syntax"
    },
    "pending_patch_summary": {
      "origin": "{\"cgm_response\": {\"model\": \"qwen3-235b-a22b-thinking-2507\", \"output_format\": \"json_patch\", \"raw_preview\": \"{\\n  \\\"summary\\\": \\\"Replace empty condition SQL in When.as_sql with '1=1' to avoid invalid CASE syntax\\\",\\n  \\\"edits\\\": [\\n    {\\n      \\\"path\\\": \\\"django/db/models/expressions.py\\\",\\n      \\\"start\\\": 1301,\\n      \\\"end\\\": 1303,\\n      \\\"new_text\\\": \\\"        condition_sql, condition_params = compiler.compile(self.condition)\\\\n        if not condition_sql.strip():\\\\n            condition_sql = '1=1'\\\\n            condition_params = []\\\\n        template_params[\\\\\\\"condition\\\\\\\"] = condition_sql\\\\n        sql_params.extend(condition_params)\\\"\\n    }\\n  ]\\n}\", \"reasoning_chars\": 31020, \"reasoning_preview\": \"We are given an issue: Case() crashes when using ~Q(pk__in=[]). The problem is that the condition in the When clause becomes empty, leading to SQL like \\\"CASE WHEN THEN ...\\\", which is invalid.\\n\\nLooking at the provided code:\\n\\nIn `When.as_sql` (lines 1297-1309):\\n```python\\n1297:     def as_sql(self, compiler, connection, template=None, **extra_context):\\n1298:         connection.ops.check_expression_support(self)\\n1299:         template_params = extra_context\\n1300:         sql_params = []\\n1301:         condition_sql, condition_params = compiler.compile(self.condition)\\n1302:         template_params[\\\"condition\\\"] = condition_sql\\n1303:         sql_params.extend(condition_params)\\n1304:         result_sql, result_params = compiler.compile(self.result)\\n1305:         template_params[\\\"result\\\"] = result_sql\\n1306:         sql_params.extend(result_params)\\n1307:         template = template or self.template\\n1308:         return template % template_params, sql_params\\n```\\n\\nThe issue is that when `self.condition` is `~Q(pk__in=[])`, the compiled `condition_sql`...<truncated 1089 chars>",
      "patch_preview": {
        "edit_count": 1,
        "edits": [
          {
            "end": 1303,
            "new_text": "condition_sql, condition_params = compiler.compile(self.condition)\n        if not condition_sql.strip():\n            condition_sql = '1=1'\n            condition_params = []\n        template_params[\"condition\"] = condition_sql\n        sql_params.extend(condition_params)",
            "path": "django/db/models/expressions.py",
            "start": 1301
          }
        ],
        "summary": "Replace empty condition SQL in When.as_sql with '1=1' to avoid invalid CASE syntax"
      },
      "summary": "Replace empty condition SQL in When.as_sql with '1=1' to avoid invalid CASE syntax",
      "touched_paths": [
        "django/db/models/expressions.py"
      ]
    },
    "rolled_back": true,
    "source_tree_state": "unchanged_pending_patch_saved",
    "status": "patch_proposed",
    "summary": "Replace empty condition SQL in When.as_sql with '1=1' to avoid invalid CASE syntax",
    "tool": "repair_propose",
    "touched_paths": [
      "django/db/models/expressions.py"
    ]
  },
  "step": 10,
  "visible_thinking": "The issue is that when a `When` clause compiles to an empty SQL string (such as with `~Q(pk__in=[])`), the `Case.as_sql` method includes this empty string in the `case_parts` list. When these parts are joined, it results in malformed SQL like `CASE WHEN THEN ...`, which causes a syntax error.\nFrom the previous attempt, the patch tried to skip empty `case_sql` strings by checking `if not case_sql.strip(): continue`. However, this approach was not sufficient, as the test still failed. The error message indicates that the SQL generated still contains `WHEN THEN`, meaning the empty condition is not being handled correctly.\nLet's re-examine the `When.as_sql` method to understand how it generates the SQL and why it might produce an empty string. The `When` class's `as_sql` method uses a template `\"WHEN %(condition)s THEN %(result)s\"`. If `condition_sql` is empty, the resulting SQL will be `WHEN THEN ...`, which is invalid.\nThe problem likely lies in the `When.as_sql` method where `condition_sql` is compiled. If the condition resolves to an empty result set, it should either raise `EmptyResultSet` or handle it in a way that prevents the `WHEN` clause from being included in the `Case` statement.\nSince the previous patch on `Case.as_sql` didn't work, I should focus on the `When.as_sql` method to ensure it doesn't return a malformed SQL fragment when the condition is empty. Perhaps the `When.as_sql` method should check if `condition_sql` is empty and raise `EmptyResultSet` in that case, so the `Case.as_sql` method can catch it and skip that `When` clause.\nLet's look at the `When.as_sql` method again. It compiles `self.condition` and `self.result`. If `condition_sql` is empty, it should not proceed to create the `WHEN ... THEN ...` string. Instead, it should signal that this `When` clause should be skipped.\nI think the correct approach is to modify `When.as_sql` to detect when `condition_sql` is empty and raise `EmptyResultSet` in that scenario. This way, the existing logic in `Case.as_sql` that catches `EmptyResultSet` will handle it appropriately.\nLet's prepare a repair proposal for the `When.as_sql` method."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "decision": "The pending patch correctly addresses the issue by replacing empty condition SQL with '1=1' in When.as_sql, which prevents the invalid 'WHEN THEN' syntax and ensures all rows match as expected when using ~Q(pk__in=[])."
    },
    "tool": "repair_submit"
  },
  "result": {
    "cgm_response": {
      "model": "qwen3-235b-a22b-thinking-2507",
      "output_format": "json_patch",
      "raw_preview": "{\n  \"summary\": \"Replace empty condition SQL in When.as_sql with '1=1' to avoid invalid CASE syntax\",\n  \"edits\": [\n    {\n      \"path\": \"django/db/models/expressions.py\",\n      \"start\": 1301,\n      \"end\": 1303,\n      \"new_text\": \"        condition_sql, condition_params = compiler.compile(self.condition)\\n        if not condition_sql.strip():\\n            condition_sql = '1=1'\\n            condition_params = []\\n        template_params[\\\"condition\\\"] = condition_sql\\n        sql_params.extend(condition_params)\"\n    }\n  ]\n}",
      "reasoning_chars": 31020,
      "reasoning_preview": "We are given an issue: Case() crashes when using ~Q(pk__in=[]). The problem is that the condition in the When clause becomes empty, leading to SQL like \"CASE WHEN THEN ...\", which is invalid.\n\nLooking at the provided code:\n\nIn `When.as_sql` (lines 1297-1309):\n```python\n1297:     def as_sql(self, compiler, connection, template=None, **extra_context):\n1298:         connection.ops.check_expression_support(self)\n1299:         template_params = extra_context\n1300:         sql_params = []\n1301:         condition_sql, condition_params = compiler.compile(self.condition)\n1302:         template_params[\"condition\"] = condition_sql\n1303:         sql_params.extend(condition_params)\n1304:         result_sql, result_params = compiler.compile(self.result)\n1305:         template_params[\"result\"] = result_sql\n1306:         sql_params.extend(result_params)\n1307:         template = template or self.template\n1308:         return template % template_params, sql_params\n```\n\nThe issue is that when `self.condition` is `~Q(pk__in=[])`, the compiled `condition_sql` becomes an empty string. Then, the template `\"WHEN %(condition)s THEN %(result)s\"` becomes `\"WHEN  THEN ...\"` which is invalid.\n\nThe planner guid",
      "thinking_enabled": true
    },
    "done": true,
    "patch_preview": {
      "edit_count": 1,
      "edits": [
        {
          "end": 1303,
          "new_text": "condition_sql, condition_params = compiler.compile(self.condition)\n        if not condition_sql.strip():\n            condition_sql = '1=1'\n            condition_params = []\n        template_params[\"condition\"] = condition_sql\n        sql_params.extend(condition_params)",
          "path": "django/db/models/expressions.py",
          "start": 1301
        }
      ],
      "summary": "Replace empty condition SQL in When.as_sql with '1=1' to avoid invalid CASE syntax"
    },
    "rolled_back": false,
    "status": "passed",
    "submit_decision": "The pending patch correctly addresses the issue by replacing empty condition SQL with '1=1' in When.as_sql, which prevents the invalid 'WHEN THEN' syntax and ensures all rows match as expected when using ~Q(pk__in=[]).",
    "summary": "Replace empty condition SQL in When.as_sql with '1=1' to avoid invalid CASE syntax",
    "test_summary": {
      "command": "<official SWE-bench eval command omitted; contains benchmark harness setup/test patch>",
      "command_omitted_for_benchmark_hygiene": true,
      "excerpt": "Actual runtime failure summary (benchmark expected values omitted):",
      "implementation_frames": [],
      "parser_error": "missing version for SWE-bench log parser",
      "resolved": null,
      "returncode": 0,
      "runtime_observations": {
        "actual_assertion_values": [],
        "actual_messages": [],
        "exception_types": [],
        "failed_selectors": [],
        "omitted_hidden_expected_values": false,
        "policy": "actual runtime output only; benchmark test source and hidden expected values are omitted"
      },
      "status": "passed",
      "tests_status": {}
    },
    "tool": "repair_submit",
    "touched_paths": [
      "django/db/models/expressions.py"
    ]
  },
  "step": 11,
  "visible_thinking": "Let me analyze the current situation:\n\n1. There's a pending patch proposed that modifies `When.as_sql` to replace empty condition SQL with '1=1'\n2. The error is \"near 'THEN': syntax error\" which occurs when `~Q(pk__in=[])` compiles to an empty condition\n3. Looking at the proposed patch, it's trying to handle the case where `condition_sql` is empty by replacing it with '1=1' (which evaluates to true)\n\nHowever, I need to examine this more carefully. The issue is that when `~Q(pk__in=[])` is compiled, it produces an empty string, which when plugged into the template `\"WHEN %(condition)s THEN %(result)s\"` creates `\"WHEN THEN ...\"`, which is invalid SQL.\n\nThe proposed solution replaces empty condition with '1=1', which would make it `\"WHEN 1=1 THEN ...\"` which is valid SQL and always evaluates to true (matching all rows).\n\nBut let me check if this approach is correct by looking at the current code in `When.as_sql`. From the working code, I can see the current implementation:\n\n```python\ndef as_sql(self, compiler, connection, template=None, **extra_context):\n    connection.ops.check_expression_support(self)\n    template_params = extra_context\n    sql_params = []\n    condition_sql, condition_params = compiler.compile(self.condition)\n    template_params[\"condition\"] = condition_sql\n    sql_params.extend(condition_params)\n    result_sql, result_params = compiler.compile(self.result)\n    template_params[\"result\"] = result_sql\n    sql_params.extend(result_params)\n    template = template or self.template\n    return template % template_params, sql_params\n```\n\nThe proposed patch modifies lines 1301-1303 to add a check for empty condition SQL and replace it with '1=1'.\n\nThis seems like a reasonable approach. Let me verify that this would solve the issue:\n- When `~Q(pk__in=[])` compiles to an empty string\n- The patch detects this and replaces it with '1=1'\n- The resulting SQL becomes `WHEN 1=1 THEN ...` which is valid and matches all rows\n- This matches the expected behavior described in the issue\n\nThe pending patch looks correct. Let me submit it for testing."
}
```
