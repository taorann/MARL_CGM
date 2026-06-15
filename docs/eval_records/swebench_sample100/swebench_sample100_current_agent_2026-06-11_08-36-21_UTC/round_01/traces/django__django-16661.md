
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
      "excerpt": "Actual runtime failure summary (benchmark expected values omitted):\nexception_types: AssertionError\nactual_messages:\n- False is not True",
      "implementation_frames": [],
      "parser_error": "missing version for SWE-bench log parser",
      "resolved": null,
      "returncode": 1,
      "runtime_observations": {
        "actual_assertion_values": [],
        "actual_messages": [
          "False is not True"
        ],
        "exception_types": [
          "AssertionError"
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
  "step": 1,
  "visible_thinking": "The issue describes a problem with `ModelAdmin.lookup_allowed()` incorrectly raising a `DisallowedModelAdminLookup` when a foreign key is used as a primary key. The issue seems to be related to how the admin interprets foreign key relationships, particularly when a `OneToOneField` is used as a primary key. The test case provided demonstrates this issue.\nSince I don't have any runtime behavior or code read yet, I need to start by running the failed test to understand the behavior. This will help me gather the necessary information to proceed with the repair."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "find_type": "class",
      "path_glob": "**/*.py",
      "query": "ModelAdmin"
    },
    "tool": "explore_find"
  },
  "result": {
    "path_glob": "**/*.py",
    "result_policy": "function/class/method/assignment results include a small implementation preview in W for orientation only; read the node before memory_commit or repair evidence. file-level results list top symbols instead of full file text",
    "results": [
      {
        "code": " 621: class ModelAdmin(BaseModelAdmin):\n 622:     \"\"\"Encapsulate all admin options and functionality for a given model.\"\"\"\n 623: \n 624:     list_display = (\"__str__\",)\n 625:     list_display_links = ()\n 626:     list_filter = ()\n 627:     list_select_related = False\n 628:     list_per_page = 100\n 629:     list_max_show_all = 200\n 630:     list_editable = ()\n 631:     search_fields = ()\n 632:     search_help_text = None\n 633:     date_hierarchy = None\n 634:     save_as = False\n 635:     save_as_continue = True\n 636:     save_on_top = False\n 637:     paginator = Paginator\n 638:     preserve_filters = True\n 639:     show_facets = ShowFacets.ALLOW\n 640:     inlines = ()\n 641: \n 642:     # Custom templates (designed to be over-ridden in subclasses)\n 643:     add_form_template = None\n 644:     change_form_template = None\n 645:     change_list_template = None\n 646:     delete_confirmation_template = None\n 647:     delete_selected_confirmation_template = None\n 648:     object_history_template = None\n 649:     popup_response_template = None\n 650: ",
        "code_preview_lines": [
          621,
          650
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/admin/options.py:ModelAdmin:621",
        "kind": "class",
        "lines": [
          621,
          2294
        ],
        "local_symbol_references": [
          {
            "appearance_index": 17,
            "id": "class:django/contrib/admin/options.py:BaseModelAdmin:124",
            "kind": "class",
            "lines": [
              124,
              618
            ],
            "name": "BaseModelAdmin",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "BaseModelAdmin"
          },
          {
            "appearance_index": 521,
            "id": "class:django/contrib/admin/options.py:ShowFacets:75",
            "kind": "class",
            "lines": [
              75,
              78
            ],
            "name": "ShowFacets",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "ShowFacets"
          }
        ],
        "name": "ModelAdmin",
        "path": "django/contrib/admin/options.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 17,
            "id": "class:django/contrib/admin/options.py:BaseModelAdmin:124",
            "kind": "class",
            "lines": [
              124,
              618
            ],
            "name": "BaseModelAdmin",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "BaseModelAdmin"
          },
          {
            "appearance_index": 521,
            "id": "class:django/contrib/admin/options.py:ShowFacets:75",
            "kind": "class",
            "lines": [
              75,
              78
            ],
            "name": "ShowFacets",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "ShowFacets"
          }
        ]
      },
      {
        "code": " 176: class BaseModelAdminChecks:\n 177:     def check(self, admin_obj, **kwargs):\n 178:         return [\n 179:             *self._check_autocomplete_fields(admin_obj),\n 180:             *self._check_raw_id_fields(admin_obj),\n 181:             *self._check_fields(admin_obj),\n 182:             *self._check_fieldsets(admin_obj),\n 183:             *self._check_exclude(admin_obj),\n 184:             *self._check_form(admin_obj),\n 185:             *self._check_filter_vertical(admin_obj),\n 186:             *self._check_filter_horizontal(admin_obj),\n 187:             *self._check_radio_fields(admin_obj),\n 188:             *self._check_prepopulated_fields(admin_obj),\n 189:             *self._check_view_on_site_url(admin_obj),\n 190:             *self._check_ordering(admin_obj),\n 191:             *self._check_readonly_fields(admin_obj),\n 192:         ]\n 193: \n 194:     def _check_autocomplete_fields(self, obj):\n 195:         \"\"\"\n 196:         Check that `autocomplete_fields` is a list or tuple of model fields.\n 197:         \"\"\"\n 198:         if not isinstance(obj.autocomplete_fields, (list, tuple)):\n 199:             return must_be(\n 200:                 \"a list or tuple\",\n 201:                 option=\"autocomplete_fields\",\n 202:                 obj=obj,\n 203:                 id=\"admin.E036\",\n 204:             )\n 205:         else:",
        "code_preview_lines": [
          176,
          205
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/admin/checks.py:BaseModelAdminChecks:176",
        "kind": "class",
        "lines": [
          176,
          786
        ],
        "local_symbol_references": [
          {
            "appearance_index": 36,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks.check:177",
            "kind": "function",
            "lines": [
              177,
              192
            ],
            "name": "BaseModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 36,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks.check:790",
            "kind": "function",
            "lines": [
              790,
              807
            ],
            "name": "ModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 36,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks.check:1231",
            "kind": "function",
            "lines": [
              1231,
              1241
            ],
            "name": "InlineModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 105,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks._check_autocomplete_fields:194",
            "kind": "function",
            "lines": [
              194,
              215
            ],
            "name": "BaseModelAdminChecks._check_autocomplete_fields",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_autocomplete_fields"
          },
          {
            "appearance_index": 162,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks._check_raw_id_fields:266",
            "kind": "function",
            "lines": [
              266,
              282
            ],
            "name": "BaseModelAdminChecks._check_raw_id_fields",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_raw_id_fields"
          },
          {
            "appearance_index": 213,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks._check_fields:314",
            "kind": "function",
            "lines": [
              314,
              346
            ],
            "name": "BaseModelAdminChecks._check_fields",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_fields"
          },
          {
            "appearance_index": 257,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks._check_fieldsets:348",
            "kind": "function",
            "lines": [
              348,
              367
            ],
            "name": "BaseModelAdminChecks._check_fieldsets",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_fieldsets"
          },
          {
            "appearance_index": 304,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks._check_exclude:460",
            "kind": "function",
            "lines": [
              460,
              478
            ],
            "name": "BaseModelAdminChecks._check_exclude",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_exclude"
          }
        ],
        "name": "BaseModelAdminChecks",
        "path": "django/contrib/admin/checks.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 36,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks.check:177",
            "kind": "function",
            "lines": [
              177,
              192
            ],
            "name": "BaseModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 36,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks.check:790",
            "kind": "function",
            "lines": [
              790,
              807
            ],
            "name": "ModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 36,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks.check:1231",
            "kind": "function",
            "lines": [
              1231,
              1241
            ],
            "name": "InlineModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 105,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks._check_autocomplete_fields:194",
            "kind": "function",
            "lines": [
              194,
              215
            ],
            "name": "BaseModelAdminChecks._check_autocomplete_fields",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_autocomplete_fields"
          },
          {
            "appearance_index": 162,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks._check_raw_id_fields:266",
            "kind": "function",
            "lines": [
              266,
              282
            ],
            "name": "BaseModelAdminChecks._check_raw_id_fields",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_raw_id_fields"
          },
          {
            "appearance_index": 213,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks._check_fields:314",
            "kind": "function",
            "lines": [
              314,
              346
            ],
            "name": "BaseModelAdminChecks._check_fields",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_fields"
          },
          {
            "appearance_index": 257,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks._check_fieldsets:348",
            "kind": "function",
            "lines": [
              348,
              367
            ],
            "name": "BaseModelAdminChecks._check_fieldsets",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_fieldsets"
          },
          {
            "appearance_index": 304,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks._check_exclude:460",
            "kind": "function",
            "lines": [
              460,
              478
            ],
            "name": "BaseModelAdminChecks._check_exclude",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_exclude"
          }
        ]
      },
      {
        "code": "1230: class InlineModelAdminChecks(BaseModelAdminChecks):\n1231:     def check(self, inline_obj, **kwargs):\n1232:         parent_model = inline_obj.parent_model\n1233:         return [\n1234:             *super().check(inline_obj),\n1235:             *self._check_relation(inline_obj, parent_model),\n1236:             *self._check_exclude_of_parent_model(inline_obj, parent_model),\n1237:             *self._check_extra(inline_obj),\n1238:             *self._check_max_num(inline_obj),\n1239:             *self._check_min_num(inline_obj),\n1240:             *self._check_formset(inline_obj),\n1241:         ]\n1242: \n1243:     def _check_exclude_of_parent_model(self, obj, parent_model):\n1244:         # Do not perform more specific checks if the base checks result in an\n1245:         # error.\n1246:         errors = super()._check_exclude(obj)\n1247:         if errors:\n1248:             return []\n1249: \n1250:         # Skip if `fk_name` is invalid.\n1251:         if self._check_relation(obj, parent_model):\n1252:             return []\n1253: \n1254:         if obj.exclude is None:\n1255:             return []\n1256: \n1257:         fk = _get_foreign_key(parent_model, obj.model, fk_name=obj.fk_name)\n1258:         if fk.name in obj.exclude:\n1259:             return [",
        "code_preview_lines": [
          1230,
          1259
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/admin/checks.py:InlineModelAdminChecks:1230",
        "kind": "class",
        "lines": [
          1230,
          1318
        ],
        "local_symbol_references": [
          {
            "appearance_index": 29,
            "id": "class:django/contrib/admin/checks.py:BaseModelAdminChecks:176",
            "kind": "class",
            "lines": [
              176,
              786
            ],
            "name": "BaseModelAdminChecks",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "BaseModelAdminChecks"
          },
          {
            "appearance_index": 60,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks.check:177",
            "kind": "function",
            "lines": [
              177,
              192
            ],
            "name": "BaseModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 60,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks.check:790",
            "kind": "function",
            "lines": [
              790,
              807
            ],
            "name": "ModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 60,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks.check:1231",
            "kind": "function",
            "lines": [
              1231,
              1241
            ],
            "name": "InlineModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 217,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks._check_relation:1274",
            "kind": "function",
            "lines": [
              1274,
              1280
            ],
            "name": "InlineModelAdminChecks._check_relation",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_relation"
          },
          {
            "appearance_index": 278,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks._check_exclude_of_parent_model:1243",
            "kind": "function",
            "lines": [
              1243,
              1272
            ],
            "name": "InlineModelAdminChecks._check_exclude_of_parent_model",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_exclude_of_parent_model"
          },
          {
            "appearance_index": 354,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks._check_extra:1282",
            "kind": "function",
            "lines": [
              1282,
              1288
            ],
            "name": "InlineModelAdminChecks._check_extra",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_extra"
          },
          {
            "appearance_index": 398,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks._check_max_num:1290",
            "kind": "function",
            "lines": [
              1290,
              1298
            ],
            "name": "InlineModelAdminChecks._check_max_num",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_max_num"
          }
        ],
        "name": "InlineModelAdminChecks",
        "path": "django/contrib/admin/checks.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 29,
            "id": "class:django/contrib/admin/checks.py:BaseModelAdminChecks:176",
            "kind": "class",
            "lines": [
              176,
              786
            ],
            "name": "BaseModelAdminChecks",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "BaseModelAdminChecks"
          },
          {
            "appearance_index": 60,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks.check:177",
            "kind": "function",
            "lines": [
              177,
              192
            ],
            "name": "BaseModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 60,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks.check:790",
            "kind": "function",
            "lines": [
              790,
              807
            ],
            "name": "ModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 60,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks.check:1231",
            "kind": "function",
            "lines": [
              1231,
              1241
            ],
            "name": "InlineModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 217,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks._check_relation:1274",
            "kind": "function",
            "lines": [
              1274,
              1280
            ],
            "name": "InlineModelAdminChecks._check_relation",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_relation"
          },
          {
            "appearance_index": 278,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks._check_exclude_of_parent_model:1243",
            "kind": "function",
            "lines": [
              1243,
              1272
            ],
            "name": "InlineModelAdminChecks._check_exclude_of_parent_model",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_exclude_of_parent_model"
          },
          {
            "appearance_index": 354,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks._check_extra:1282",
            "kind": "function",
            "lines": [
              1282,
              1288
            ],
            "name": "InlineModelAdminChecks._check_extra",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_extra"
          },
          {
            "appearance_index": 398,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks._check_max_num:1290",
            "kind": "function",
            "lines": [
              1290,
              1298
            ],
            "name": "InlineModelAdminChecks._check_max_num",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_max_num"
          }
        ]
      },
      {
        "code": " 789: class ModelAdminChecks(BaseModelAdminChecks):\n 790:     def check(self, admin_obj, **kwargs):\n 791:         return [\n 792:             *super().check(admin_obj),\n 793:             *self._check_save_as(admin_obj),\n 794:             *self._check_save_on_top(admin_obj),\n 795:             *self._check_inlines(admin_obj),\n 796:             *self._check_list_display(admin_obj),\n 797:             *self._check_list_display_links(admin_obj),\n 798:             *self._check_list_filter(admin_obj),\n 799:             *self._check_list_select_related(admin_obj),\n 800:             *self._check_list_per_page(admin_obj),\n 801:             *self._check_list_max_show_all(admin_obj),\n 802:             *self._check_list_editable(admin_obj),\n 803:             *self._check_search_fields(admin_obj),\n 804:             *self._check_date_hierarchy(admin_obj),\n 805:             *self._check_action_permission_methods(admin_obj),\n 806:             *self._check_actions_uniqueness(admin_obj),\n 807:         ]\n 808: \n 809:     def _check_save_as(self, obj):\n 810:         \"\"\"Check save_as is a boolean.\"\"\"\n 811: \n 812:         if not isinstance(obj.save_as, bool):\n 813:             return must_be(\"a boolean\", option=\"save_as\", obj=obj, id=\"admin.E101\")\n 814:         else:\n 815:             return []\n 816: \n 817:     def _check_save_on_top(self, obj):",
        "code_preview_lines": [
          789,
          817
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/admin/checks.py:ModelAdminChecks:789",
        "kind": "class",
        "lines": [
          789,
          1227
        ],
        "local_symbol_references": [
          {
            "appearance_index": 23,
            "id": "class:django/contrib/admin/checks.py:BaseModelAdminChecks:176",
            "kind": "class",
            "lines": [
              176,
              786
            ],
            "name": "BaseModelAdminChecks",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "BaseModelAdminChecks"
          },
          {
            "appearance_index": 54,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks.check:177",
            "kind": "function",
            "lines": [
              177,
              192
            ],
            "name": "BaseModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 54,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks.check:790",
            "kind": "function",
            "lines": [
              790,
              807
            ],
            "name": "ModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 54,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks.check:1231",
            "kind": "function",
            "lines": [
              1231,
              1241
            ],
            "name": "InlineModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 162,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks._check_save_as:809",
            "kind": "function",
            "lines": [
              809,
              815
            ],
            "name": "ModelAdminChecks._check_save_as",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_save_as"
          },
          {
            "appearance_index": 207,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks._check_save_on_top:817",
            "kind": "function",
            "lines": [
              817,
              823
            ],
            "name": "ModelAdminChecks._check_save_on_top",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_save_on_top"
          },
          {
            "appearance_index": 256,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks._check_inlines:825",
            "kind": "function",
            "lines": [
              825,
              838
            ],
            "name": "ModelAdminChecks._check_inlines",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_inlines"
          },
          {
            "appearance_index": 301,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks._check_list_display:878",
            "kind": "function",
            "lines": [
              878,
              891
            ],
            "name": "ModelAdminChecks._check_list_display",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_list_display"
          }
        ],
        "name": "ModelAdminChecks",
        "path": "django/contrib/admin/checks.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 23,
            "id": "class:django/contrib/admin/checks.py:BaseModelAdminChecks:176",
            "kind": "class",
            "lines": [
              176,
              786
            ],
            "name": "BaseModelAdminChecks",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "BaseModelAdminChecks"
          },
          {
            "appearance_index": 54,
            "id": "func:django/contrib/admin/checks.py:BaseModelAdminChecks.check:177",
            "kind": "function",
            "lines": [
              177,
              192
            ],
            "name": "BaseModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 54,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks.check:790",
            "kind": "function",
            "lines": [
              790,
              807
            ],
            "name": "ModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 54,
            "id": "func:django/contrib/admin/checks.py:InlineModelAdminChecks.check:1231",
            "kind": "function",
            "lines": [
              1231,
              1241
            ],
            "name": "InlineModelAdminChecks.check",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 162,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks._check_save_as:809",
            "kind": "function",
            "lines": [
              809,
              815
            ],
            "name": "ModelAdminChecks._check_save_as",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_save_as"
          },
          {
            "appearance_index": 207,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks._check_save_on_top:817",
            "kind": "function",
            "lines": [
              817,
              823
            ],
            "name": "ModelAdminChecks._check_save_on_top",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_save_on_top"
          },
          {
            "appearance_index": 256,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks._check_inlines:825",
            "kind": "function",
            "lines": [
              825,
              838
            ],
            "name": "ModelAdminChecks._check_inlines",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_inlines"
          },
          {
            "appearance_index": 301,
            "id": "func:django/contrib/admin/checks.py:ModelAdminChecks._check_list_display:878",
            "kind": "function",
            "lines": [
              878,
              891
            ],
            "name": "ModelAdminChecks._check_list_display",
            "path": "django/contrib/admin/checks.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "_check_list_display"
          }
        ]
      },
      {
        "code": " 124: class BaseModelAdmin(metaclass=forms.MediaDefiningClass):\n 125:     \"\"\"Functionality common to both ModelAdmin and InlineAdmin.\"\"\"\n 126: \n 127:     autocomplete_fields = ()\n 128:     raw_id_fields = ()\n 129:     fields = None\n 130:     exclude = None\n 131:     fieldsets = None\n 132:     form = forms.ModelForm\n 133:     filter_vertical = ()\n 134:     filter_horizontal = ()\n 135:     radio_fields = {}\n 136:     prepopulated_fields = {}\n 137:     formfield_overrides = {}\n 138:     readonly_fields = ()\n 139:     ordering = None\n 140:     sortable_by = None\n 141:     view_on_site = True\n 142:     show_full_result_count = True\n 143:     checks_class = BaseModelAdminChecks\n 144: \n 145:     def check(self, **kwargs):\n 146:         return self.checks_class().check(self, **kwargs)\n 147: \n 148:     def __init__(self):\n 149:         # Merge FORMFIELD_FOR_DBFIELD_DEFAULTS with the formfield_overrides\n 150:         # rather than simply overwriting.\n 151:         overrides = copy.deepcopy(FORMFIELD_FOR_DBFIELD_DEFAULTS)\n 152:         for k, v in self.formfield_overrides.items():\n 153:             overrides.setdefault(k, {}).update(v)",
        "code_preview_lines": [
          124,
          153
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/admin/options.py:BaseModelAdmin:124",
        "kind": "class",
        "lines": [
          124,
          618
        ],
        "local_symbol_references": [
          {
            "appearance_index": 94,
            "id": "class:django/contrib/admin/options.py:ModelAdmin:621",
            "kind": "class",
            "lines": [
              621,
              2294
            ],
            "name": "ModelAdmin",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "ModelAdmin"
          },
          {
            "appearance_index": 570,
            "id": "func:django/contrib/admin/options.py:BaseModelAdmin.check:145",
            "kind": "function",
            "lines": [
              145,
              146
            ],
            "name": "BaseModelAdmin.check",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 691,
            "id": "module_assignment:django/contrib/admin/options.py:FORMFIELD_FOR_DBFIELD_DEFAULTS:103",
            "kind": "assignment",
            "lines": [
              103,
              119
            ],
            "name": "FORMFIELD_FOR_DBFIELD_DEFAULTS",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "FORMFIELD_FOR_DBFIELD_DEFAULTS"
          }
        ],
        "name": "BaseModelAdmin",
        "path": "django/contrib/admin/options.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 94,
            "id": "class:django/contrib/admin/options.py:ModelAdmin:621",
            "kind": "class",
            "lines": [
              621,
              2294
            ],
            "name": "ModelAdmin",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "ModelAdmin"
          },
          {
            "appearance_index": 570,
            "id": "func:django/contrib/admin/options.py:BaseModelAdmin.check:145",
            "kind": "function",
            "lines": [
              145,
              146
            ],
            "name": "BaseModelAdmin.check",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "check"
          },
          {
            "appearance_index": 691,
            "id": "module_assignment:django/contrib/admin/options.py:FORMFIELD_FOR_DBFIELD_DEFAULTS:103",
            "kind": "assignment",
            "lines": [
              103,
              119
            ],
            "name": "FORMFIELD_FOR_DBFIELD_DEFAULTS",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "FORMFIELD_FOR_DBFIELD_DEFAULTS"
          }
        ]
      },
      {
        "code": "2391:         class DeleteProtectedModelForm(base_model_form):\n2392:             def hand_clean_DELETE(self):\n2393:                 \"\"\"\n2394:                 We don't validate the 'DELETE' field itself because on\n2395:                 templates it's not rendered using the field information, but\n2396:                 just using a generic \"deletion_field\" of the InlineModelAdmin.\n2397:                 \"\"\"\n2398:                 if self.cleaned_data.get(DELETION_FIELD_NAME, False):\n2399:                     using = router.db_for_write(self._meta.model)\n2400:                     collector = NestedObjects(using=using)\n2401:                     if self.instance._state.adding:\n2402:                         return\n2403:                     collector.collect([self.instance])\n2404:                     if collector.protected:\n2405:                         objs = []\n2406:                         for p in collector.protected:\n2407:                             objs.append(\n2408:                                 # Translators: Model verbose name and instance\n2409:                                 # representation, suitable to be an item in a\n2410:                                 # list.\n2411:                                 _(\"%(class_name)s %(instance)s\")",
        "code_preview_lines": [
          2391,
          2411
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/admin/options.py:InlineModelAdmin.get_formset.DeleteProtectedModelForm:2391",
        "kind": "class",
        "lines": [
          2391,
          2439
        ],
        "local_symbol_references": [
          {
            "appearance_index": 73,
            "id": "func:django/contrib/admin/options.py:InlineModelAdmin.get_formset.DeleteProtectedModelForm.hand_clean_DELETE:2392",
            "kind": "function",
            "lines": [
              2392,
              2426
            ],
            "name": "InlineModelAdmin.get_formset.DeleteProtectedModelForm.hand_clean_DELETE",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "hand_clean_DELETE"
          },
          {
            "appearance_index": 327,
            "id": "class:django/contrib/admin/options.py:InlineModelAdmin:2297",
            "kind": "class",
            "lines": [
              2297,
              2505
            ],
            "name": "InlineModelAdmin",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "InlineModelAdmin"
          }
        ],
        "name": "InlineModelAdmin.get_formset.DeleteProtectedModelForm",
        "path": "django/contrib/admin/options.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 73,
            "id": "func:django/contrib/admin/options.py:InlineModelAdmin.get_formset.DeleteProtectedModelForm.hand_clean_DELETE:2392",
            "kind": "function",
            "lines": [
              2392,
              2426
            ],
            "name": "InlineModelAdmin.get_formset.DeleteProtectedModelForm.hand_clean_DELETE",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "hand_clean_DELETE"
          },
          {
            "appearance_index": 327,
            "id": "class:django/contrib/admin/options.py:InlineModelAdmin:2297",
            "kind": "class",
            "lines": [
              2297,
              2505
            ],
            "name": "InlineModelAdmin",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "InlineModelAdmin"
          }
        ]
      },
      {
        "code": "2297: class InlineModelAdmin(BaseModelAdmin):\n2298:     \"\"\"\n2299:     Options for inline editing of ``model`` instances.\n2300: \n2301:     Provide ``fk_name`` to specify the attribute name of the ``ForeignKey``\n2302:     from ``model`` to its parent. This is required if ``model`` has more than\n2303:     one ``ForeignKey`` to its parent.\n2304:     \"\"\"\n2305: \n2306:     model = None\n2307:     fk_name = None\n2308:     formset = BaseInlineFormSet\n2309:     extra = 3\n2310:     min_num = None\n2311:     max_num = None\n2312:     template = None\n2313:     verbose_name = None\n2314:     verbose_name_plural = None\n2315:     can_delete = True\n2316:     show_change_link = False\n2317:     checks_class = InlineModelAdminChecks\n2318:     classes = None\n2319: \n2320:     def __init__(self, parent_model, admin_site):\n2321:         self.admin_site = admin_site\n2322:         self.parent_model = parent_model\n2323:         self.opts = self.model._meta\n2324:         self.has_registered_model = admin_site.is_registered(self.model)\n2325:         super().__init__()\n2326:         if self.verbose_name_plural is None:",
        "code_preview_lines": [
          2297,
          2326
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/admin/options.py:InlineModelAdmin:2297",
        "kind": "class",
        "lines": [
          2297,
          2505
        ],
        "local_symbol_references": [
          {
            "appearance_index": 23,
            "id": "class:django/contrib/admin/options.py:BaseModelAdmin:124",
            "kind": "class",
            "lines": [
              124,
              618
            ],
            "name": "BaseModelAdmin",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "BaseModelAdmin"
          }
        ],
        "name": "InlineModelAdmin",
        "path": "django/contrib/admin/options.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 23,
            "id": "class:django/contrib/admin/options.py:BaseModelAdmin:124",
            "kind": "class",
            "lines": [
              124,
              618
            ],
            "name": "BaseModelAdmin",
            "path": "django/contrib/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "BaseModelAdmin"
          }
        ]
      },
      {
        "code": "   4: class DisallowedModelAdminLookup(SuspiciousOperation):\n   5:     \"\"\"Invalid filter was passed to admin view via URL querystring\"\"\"\n   6: \n   7:     pass",
        "code_preview_lines": [
          4,
          7
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": false,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/admin/exceptions.py:DisallowedModelAdminLookup:4",
        "kind": "class",
        "lines": [
          4,
          7
        ],
        "local_symbol_references": [],
        "name": "DisallowedModelAdminLookup",
        "path": "django/contrib/admin/exceptions.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": []
      },
      {
        "code": "  10: class DisallowedModelAdminToField(SuspiciousOperation):\n  11:     \"\"\"Invalid to_field was passed to admin view via URL query string\"\"\"\n  12: \n  13:     pass",
        "code_preview_lines": [
          10,
          13
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": false,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/admin/exceptions.py:DisallowedModelAdminToField:10",
        "kind": "class",
        "lines": [
          10,
          13
        ],
        "local_symbol_references": [],
        "name": "DisallowedModelAdminToField",
        "path": "django/contrib/admin/exceptions.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": []
      },
      {
        "code": "  20: class GISModelAdmin(GeoModelAdminMixin, ModelAdmin):\n  21:     pass",
        "code_preview_lines": [
          20,
          21
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": false,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/gis/admin/options.py:GISModelAdmin:20",
        "kind": "class",
        "lines": [
          20,
          21
        ],
        "local_symbol_references": [
          {
            "appearance_index": 20,
            "id": "class:django/contrib/gis/admin/options.py:GeoModelAdminMixin:6",
            "kind": "class",
            "lines": [
              6,
              17
            ],
            "name": "GeoModelAdminMixin",
            "path": "django/contrib/gis/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "GeoModelAdminMixin"
          }
        ],
        "name": "GISModelAdmin",
        "path": "django/contrib/gis/admin/options.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 20,
            "id": "class:django/contrib/gis/admin/options.py:GeoModelAdminMixin:6",
            "kind": "class",
            "lines": [
              6,
              17
            ],
            "name": "GeoModelAdminMixin",
            "path": "django/contrib/gis/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "GeoModelAdminMixin"
          }
        ]
      },
      {
        "code": "   6: class GeoModelAdminMixin:\n   7:     gis_widget = OSMWidget\n   8:     gis_widget_kwargs = {}\n   9: \n  10:     def formfield_for_dbfield(self, db_field, request, **kwargs):\n  11:         if isinstance(db_field, models.GeometryField) and (\n  12:             db_field.dim < 3 or self.gis_widget.supports_3d\n  13:         ):\n  14:             kwargs[\"widget\"] = self.gis_widget(**self.gis_widget_kwargs)\n  15:             return db_field.formfield(**kwargs)\n  16:         else:\n  17:             return super().formfield_for_dbfield(db_field, request, **kwargs)",
        "code_preview_lines": [
          6,
          17
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": false,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/gis/admin/options.py:GeoModelAdminMixin:6",
        "kind": "class",
        "lines": [
          6,
          17
        ],
        "local_symbol_references": [
          {
            "appearance_index": 89,
            "id": "func:django/contrib/gis/admin/options.py:GeoModelAdminMixin.formfield_for_dbfield:10",
            "kind": "function",
            "lines": [
              10,
              17
            ],
            "name": "GeoModelAdminMixin.formfield_for_dbfield",
            "path": "django/contrib/gis/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "formfield_for_dbfield"
          }
        ],
        "name": "GeoModelAdminMixin",
        "path": "django/contrib/gis/admin/options.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 89,
            "id": "func:django/contrib/gis/admin/options.py:GeoModelAdminMixin.formfield_for_dbfield:10",
            "kind": "function",
            "lines": [
              10,
              17
            ],
            "name": "GeoModelAdminMixin.formfield_for_dbfield",
            "path": "django/contrib/gis/admin/options.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "formfield_for_dbfield"
          }
        ]
      },
      {
        "code": "  91: class GenericInlineModelAdmin(InlineModelAdmin):\n  92:     ct_field = \"content_type\"\n  93:     ct_fk_field = \"object_id\"\n  94:     formset = BaseGenericInlineFormSet\n  95: \n  96:     checks_class = GenericInlineModelAdminChecks\n  97: \n  98:     def get_formset(self, request, obj=None, **kwargs):\n  99:         if \"fields\" in kwargs:\n 100:             fields = kwargs.pop(\"fields\")\n 101:         else:\n 102:             fields = flatten_fieldsets(self.get_fieldsets(request, obj))\n 103:         exclude = [*(self.exclude or []), *self.get_readonly_fields(request, obj)]\n 104:         if (\n 105:             self.exclude is None\n 106:             and hasattr(self.form, \"_meta\")\n 107:             and self.form._meta.exclude\n 108:         ):\n 109:             # Take the custom ModelForm's Meta.exclude into account only if the\n 110:             # GenericInlineModelAdmin doesn't define its own.\n 111:             exclude.extend(self.form._meta.exclude)\n 112:         exclude = exclude or None\n 113:         can_delete = self.can_delete and self.has_delete_permission(request, obj)\n 114:         defaults = {\n 115:             \"ct_field\": self.ct_field,\n 116:             \"fk_field\": self.ct_fk_field,\n 117:             \"form\": self.form,\n 118:             \"formfield_callback\": partial(self.formfield_for_dbfield, request=request),",
        "code_preview_lines": [
          91,
          118
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [],
        "dispatch_tables": [],
        "id": "class:django/contrib/contenttypes/admin.py:GenericInlineModelAdmin:91",
        "kind": "class",
        "lines": [
          91,
          135
        ],
        "local_symbol_references": [
          {
            "appearance_index": 168,
            "id": "class:django/contrib/contenttypes/admin.py:GenericInlineModelAdminChecks:16",
            "kind": "class",
            "lines": [
              16,
              88
            ],
            "name": "GenericInlineModelAdminChecks",
            "path": "django/contrib/contenttypes/admin.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "GenericInlineModelAdminChecks"
          },
          {
            "appearance_index": 207,
            "id": "func:django/contrib/contenttypes/admin.py:GenericInlineModelAdmin.get_formset:98",
            "kind": "function",
            "lines": [
              98,
              135
            ],
            "name": "GenericInlineModelAdmin.get_formset",
            "path": "django/contrib/contenttypes/admin.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "get_formset"
          }
        ],
        "name": "GenericInlineModelAdmin",
        "path": "django/contrib/contenttypes/admin.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": [
          {
            "appearance_index": 168,
            "id": "class:django/contrib/contenttypes/admin.py:GenericInlineModelAdminChecks:16",
            "kind": "class",
            "lines": [
              16,
              88
            ],
            "name": "GenericInlineModelAdminChecks",
            "path": "django/contrib/contenttypes/admin.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "GenericInlineModelAdminChecks"
          },
          {
            "appearance_index": 207,
            "id": "func:django/contrib/contenttypes/admin.py:GenericInlineModelAdmin.get_formset:98",
            "kind": "function",
            "lines": [
              98,
              135
            ],
            "name": "GenericInlineModelAdmin.get_formset",
            "path": "django/contrib/contenttypes/admin.py",
            "read_status": "unread",
            "relation": "symbol_reference",
            "source": "referenced in the current implementation snippet",
            "symbol": "get_formset"
          }
        ]
      }
    ],
    "tool": "explore_find",
    "warning": null
  },
  "step": 2,
  "visible_thinking": "The issue involves `ModelAdmin.lookup_allowed()` incorrectly raising a `DisallowedModelAdminLookup` when a foreign key is used as a primary key. The problem seems to stem from the admin confusing a foreign key primary key with concrete inheritance, leading to incorrect lookup path resolution.\nFirst, I need to locate the implementation of `ModelAdmin.lookup_allowed()` to understand its logic. Then, I'll examine how it processes lookup paths involving foreign keys and primary keys. Since the test involves filtering on `restaurant__place__country`, I suspect the issue lies in how Django interprets this chain when `place` is the primary key of `Restaurant`.\nLet me start by finding the `ModelAdmin` class and specifically the `lookup_allowed` method."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "node_id": "class:django/contrib/admin/options.py:ModelAdmin:621",
      "view": "body"
    },
    "tool": "read"
  },
  "result": {
    "code": " 621: class ModelAdmin(BaseModelAdmin):\n 622:     \"\"\"Encapsulate all admin options and functionality for a given model.\"\"\"\n 623: \n 624:     list_display = (\"__str__\",)\n 625:     list_display_links = ()\n 626:     list_filter = ()\n 627:     list_select_related = False\n 628:     list_per_page = 100\n 629:     list_max_show_all = 200\n 630:     list_editable = ()\n 631:     search_fields = ()\n 632:     search_help_text = None\n 633:     date_hierarchy = None\n 634:     save_as = False\n 635:     save_as_continue = True\n 636:     save_on_top = False\n 637:     paginator = Paginator\n 638:     preserve_filters = True\n 639:     show_facets = ShowFacets.ALLOW\n 640:     inlines = ()\n 641: \n 642:     # Custom templates (designed to be over-ridden in subclasses)\n 643:     add_form_template = None\n 644:     change_form_template = None\n 645:     change_list_template = None\n 646:     delete_confirmation_template = None\n 647:     delete_selected_confirmation_template = None\n 648:     object_history_template = None\n 649:     popup_response_template = None\n 650: \n 651:     # Actions\n 652:     actions = ()\n 653:     action_form = helpers.ActionForm\n 654:     actions_on_top = True\n 655:     actions_on_bottom = False\n 656:     actions_selection_counter = True\n 657:     checks_class = ModelAdminChecks\n 658: \n 659:     def __init__(self, model, admin_site):\n 660:         self.model = model\n 661:         self.opts = model._meta\n 662:         self.admin_site = admin_site\n 663:         super().__init__()\n 664: \n 665:     def __str__(self):\n 666:         return \"%s.%s\" % (self.opts.app_label, self.__class__.__name__)\n 667: \n 668:     def __repr__(self):\n 669:         return (\n 670:             f\"<{self.__class__.__qualname__}: model={self.model.__qualname__} \"\n 671:             f\"site={self.admin_site!r}>\"\n 672:         )\n 673: \n 674:     def get_inline_instances(self, request, obj=None):\n 675:         inline_instances = []\n 676:         for inline_class in self.get_inlines(request, obj):\n 677:             inline = inline_class(self.model, self.admin_site)\n 678:             if request:\n 679:                 if not (\n 680:                     inline.has_view_or_change_permission(request, obj)\n 681:                     or inline.has_add_permission(request, obj)\n 682:                     or inline.has_delete_permission(request, obj)\n 683:                 ):\n 684:                     continue\n 685:                 if not inline.has_add_permission(request, obj):\n 686:                     inline.max_num = 0\n 687:             inline_instances.append(inline)\n 688: \n 689:         return inline_instances\n 690: \n 691:     def get_urls(self):\n 692:         from django.urls import path\n 693: \n 694:         def wrap(view):\n 695:             def wrapper(*args, **kwargs):\n 696:                 return self.admin_site.admin_view(view)(*args, **kwargs)\n 697: \n 698:             wrapper.model_admin = self\n 699:             return update_wrapper(wrapper, view)\n 700: \n 701:         info = self.opts.app_label, self.opts.model_name\n 702: \n 703:         return [\n 704:             path(\"\", wrap(self.changelist_view), name=\"%s_%s_changelist\" % info),\n 705:             path(\"add/\", wrap(self.add_view), name=\"%s_%s_add\" % info),\n 706:             path(\n 707:                 \"<path:object_id>/history/\",\n 708:                 wrap(self.history_view),\n 709:                 name=\"%s_%s_history\" % info,\n 710:             ),\n 711:             path(\n 712:                 \"<path:object_id>/delete/\",\n 713:                 wrap(self.delete_view),\n 714:                 name=\"%s_%s_delete\" % info,\n 715:             ),\n 716:             path(\n 717:                 \"<path:object_id>/change/\",\n 718:                 wrap(self.change_view),\n 719:                 name=\"%s_%s_change\" % info,\n 720:             ),\n 721:             # For backwards compatibility (was the change url before 1.9)\n 722:             path(\n 723:                 \"<path:object_id>/\",\n 724:                 wrap(\n 725:                     RedirectView.as_view(\n 726:                         pattern_name=\"%s:%s_%s_change\"\n 727:                         % ((self.admin_site.name,) + info)\n 728:                     )\n 729:                 ),\n 730:             ),\n 731:         ]\n 732: \n 733:     @property\n 734:     def urls(self):\n 735:         return self.get_urls()\n 736: \n 737:     @property\n 738:     def media(self):\n 739:         extra = \"\" if settings.DEBUG else \".min\"\n 740:         js = [\n 741:             \"vendor/jquery/jquery%s.js\" % extra,\n 742:             \"jquery.init.js\",\n 743:             \"core.js\",\n 744:             \"admin/RelatedObjectLookups.js\",\n 745:             \"actions.js\",\n 746:             \"urlify.js\",\n 747:             \"prepopulate.js\",\n 748:             \"vendor/xregexp/xregexp%s.js\" % extra,\n 749:         ]\n 750:         return forms.Media(js=[\"admin/js/%s\" % url for url in js])\n 751: \n 752:     def get_model_perms(self, request):\n 753:         \"\"\"\n 754:         Return a dict of all perms for this model. This dict has the keys\n 755:         ``add``, ``change``, ``delete``, and ``view`` mapping to the True/False\n 756:         for each of those actions.\n 757:         \"\"\"\n 758:         return {\n 759:             \"add\": self.has_add_permission(request),\n 760:             \"change\": self.has_change_permission(request),\n 761:             \"delete\": self.has_delete_permission(request),\n 762:             \"view\": self.has_view_permission(request),\n 763:         }\n 764: \n 765:     def _get_form_for_get_fields(self, request, obj):\n 766:         return self.get_form(request, obj, fields=None)\n 767: \n 768:     def get_form(self, request, obj=None, change=False, **kwargs):\n 769:         \"\"\"\n 770:         Return a Form class for use in the admin add view. This is used by\n 771:         add_view and change_view.\n 772:         \"\"\"\n 773:         if \"fields\" in kwargs:\n 774:             fields = kwargs.pop(\"fields\")\n 775:         else:\n 776:             fields = flatten_fieldsets(self.get_fieldsets(request, obj))\n 777:         excluded = self.get_exclude(request, obj)\n 778:         exclude = [] if excluded is None else list(excluded)\n 779:         readonly_fields = self.get_readonly_fields(request, obj)\n 780:         exclude.extend(readonly_fields)\n 781:         # Exclude all fields if it's a change form and the user doesn't have\n 782:         # the change permission.\n 783:         if (\n 784:             change\n 785:             and hasattr(request, \"user\")\n 786:             and not self.has_change_permission(request, obj)\n 787:         ):\n 788:             exclude.extend(fields)\n 789:         if excluded is None and hasattr(self.form, \"_meta\") and self.form._meta.exclude:\n 790:             # Take the custom ModelForm's Meta.exclude into account only if the\n 791:             # ModelAdmin doesn't define its own.\n 792:             exclude.extend(self.form._meta.exclude)\n 793:         # if exclude is an empty list we pass None to be consistent with the\n 794:         # default on modelform_factory\n 795:         exclude = exclude or None\n 796: \n 797:         # Remove declared form fields which are in readonly_fields.\n 798:         new_attrs = dict.fromkeys(\n 799:             f for f in readonly_fields if f in self.form.declared_fields\n 800:         )\n 801:         form = type(self.form.__name__, (self.form,), new_attrs)\n 802: \n 803:         defaults = {\n 804:             \"form\": form,\n 805:             \"fields\": fields,\n 806:             \"exclude\": exclude,\n 807:             \"formfield_callback\": partial(self.formfield_for_dbfield, request=request),\n 808:             **kwargs,\n 809:         }\n 810: \n 811:         if defaults[\"fields\"] is None and not modelform_defines_fields(\n 812:             defaults[\"form\"]\n 813:         ):\n 814:             defaults[\"fields\"] = forms.ALL_FIELDS\n 815: \n 816:         try:\n 817:             return modelform_factory(self.model, **defaults)\n 818:         except FieldError as e:\n 819:             raise FieldError(\n 820:                 \"%s. Check fields/fieldsets/exclude attributes of class %s.\"\n 821:                 % (e, self.__class__.__name__)\n 822:             )\n 823: \n 824:     def get_changelist(self, request, **kwargs):\n 825:         \"\"\"\n 826:         Return the ChangeList class for use on the changelist page.\n 827:         \"\"\"\n 828:         from django.contrib.admin.views.main import ChangeList\n 829: \n 830:         return ChangeList\n 831: \n 832:     def get_changelist_instance(self, request):\n 833:         \"\"\"\n 834:         Return a `ChangeList` instance based on `request`. May raise\n 835:         `IncorrectLookupParameters`.\n 836:         \"\"\"\n 837:         list_display = self.get_list_display(request)\n 838:         list_display_links = self.get_list_display_links(request, list_display)\n 839:         # Add the action checkboxes if any actions are available.\n 840:         if self.get_actions(request):\n 841:             list_display = [\"action_checkbox\", *list_display]\n 842:         sortable_by = self.get_sortable_by(request)\n 843:         ChangeList = self.get_changelist(request)\n 844:         return ChangeList(\n 845:             request,\n 846:             self.model,\n 847:             list_display,\n 848:             list_display_links,\n 849:             self.get_list_filter(request),\n 850:             self.date_hierarchy,\n 851:             self.get_search_fields(request),\n 852:             self.get_list_select_related(request),\n 853:             self.list_per_page,\n 854:             self.list_max_show_all,\n 855:             self.list_editable,\n 856:             self,\n 857:             sortable_by,\n 858:             self.search_help_text,\n 859:         )\n 860: \n 861:     def get_object(self, request, object_id, from_field=None):\n 862:         \"\"\"\n 863:         Return an instance matching the field and value provided, the primary\n 864:         key is used if no field is provided. Return ``None`` if no match is\n 865:         found or the object_id fails validation.\n 866:         \"\"\"\n 867:         queryset = self.get_queryset(request)\n 868:         model = queryset.model\n 869:         field = (\n 870:             model._meta.pk if from_field is None else model._meta.get_field(from_field)\n 871:         )\n 872:         try:\n 873:             object_id = field.to_python(object_id)\n 874:             return queryset.get(**{field.name: object_id})\n 875:         except (model.DoesNotExist, ValidationError, ValueError):\n 876:             return None\n 877: \n 878:     def get_changelist_form(self, request, **kwargs):\n 879:         \"\"\"\n 880:         Return a Form class for use in the Formset on the changelist page.\n 881:         \"\"\"\n 882:         defaults = {\n 883:             \"formfield_callback\": partial(self.formfield_for_dbfield, request=request),\n 884:             **kwargs,\n 885:         }\n 886:         if defaults.get(\"fields\") is None and not modelform_defines_fields(\n 887:             defaults.get(\"form\")\n 888:         ):\n 889:             defaults[\"fields\"] = forms.ALL_FIELDS\n 890: \n 891:         return modelform_factory(self.model, **defaults)\n 892: \n 893:     def get_changelist_formset(self, request, **kwargs):\n 894:         \"\"\"\n 895:         Return a FormSet class for use on the changelist page if list_editable\n 896:         is used.\n 897:         \"\"\"\n 898:         defaults = {\n 899:             \"formfield_callback\": partial(self.formfield_for_dbfield, request=request),\n 900:             **kwargs,\n 901:         }\n 902:         return modelformset_factory(\n 903:             self.model,\n 904:             self.get_changelist_form(request),\n 905:             extra=0,\n 906:             fields=self.list_editable,\n 907:             **defaults,\n 908:         )\n 909: \n 910:     def get_formsets_with_inlines(self, request, obj=None):\n 911:         \"\"\"\n 912:         Yield formsets and the corresponding inlines.\n 913:         \"\"\"\n 914:         for inline in self.get_inline_instances(request, obj):\n 915:             yield inline.get_formset(request, obj), inline\n 916: \n 917:     def get_paginator(\n 918:         self, request, queryset, per_page, orphans=0, allow_empty_first_page=True\n 919:     ):\n 920:         return self.paginator(queryset, per_page, orphans, allow_empty_first_page)\n 921: \n 922:     def log_addition(self, request, obj, message):\n 923:         \"\"\"\n 924:         Log that an object has been successfully added.\n 925: \n 926:         The default implementation creates an admin LogEntry object.\n 927:         \"\"\"\n 928:         from django.contrib.admin.models import ADDITION, LogEntry\n 929: \n 930:         return LogEntry.objects.log_action(\n 931:             user_id=request.user.pk,\n 932:             content_type_id=get_content_type_for_model(obj).pk,\n 933:             object_id=obj.pk,\n 934:             object_repr=str(obj),\n 935:             action_flag=ADDITION,\n 936:             change_message=message,\n 937:         )\n 938: \n 939:     def log_change(self, request, obj, message):\n 940:         \"\"\"\n 941:         Log that an object has been successfully changed.\n 942: \n 943:         The default implementation creates an admin LogEntry object.\n 944:         \"\"\"\n 945:         from django.contrib.admin.models import CHANGE, LogEntry\n 946: \n 947:         return LogEntry.objects.log_action(\n 948:             user_id=request.user.pk,\n 949:             content_type_id=get_content_type_for_model(obj).pk,\n 950:             object_id=obj.pk,\n 951:             object_repr=str(obj),\n 952:             action_flag=CHANGE,\n 953:             change_message=message,\n 954:         )\n 955: \n 956:     def log_deletion(self, request, obj, object_repr):\n 957:         \"\"\"\n 958:         Log that an object will be deleted. Note that this method must be\n 959:         called before the deletion.\n 960: \n 961:         The default implementation creates an admin LogEntry object.\n 962:         \"\"\"\n 963:         from django.contrib.admin.models import DELETION, LogEntry\n 964: \n 965:         return LogEntry.objects.log_action(\n 966:             user_id=request.user.pk,\n 967:             content_type_id=get_content_type_for_model(obj).pk,\n 968:             object_id=obj.pk,\n 969:             object_repr=object_repr,\n 970:             action_flag=DELETION,\n 971:         )\n 972: \n 973:     def action_checkbox(self, obj):\n 974:         \"\"\"\n 975:         A list_display column containing a checkbox widget.\n 976:         \"\"\"\n 977:         attrs = {\n 978:             \"class\": \"action-select\",\n 979:             \"aria-label\": format_html(_(\"Select this object for an action - {}\"), obj),\n 980:         }\n 981:         checkbox = forms.CheckboxInput(attrs, lambda value: False)\n 982:         return checkbox.render(helpers.ACTION_CHECKBOX_NAME, str(obj.pk))\n 983: \n 984:     @staticmethod\n 985:     def _get_action_description(func, name):\n 986:         return getattr(func, \"short_description\", capfirst(name.replace(\"_\", \" \")))\n 987: \n 988:     def _get_base_actions(self):\n 989:         \"\"\"Return the list of actions, prior to any request-based filtering.\"\"\"\n 990:         actions = []\n 991:         base_actions = (self.get_action(action) for action in self.actions or [])\n 992:         # get_action might have returned None, so filter any of those out.\n 993:         base_actions = [action for action in base_actions if action]\n 994:         base_action_names = {name for _, name, _ in base_actions}\n 995: \n 996:         # Gather actions from the admin site first\n 997:         for name, func in self.admin_site.actions:\n 998:             if name in base_action_names:\n 999:                 continue\n1000:             description = self._get_action_description(func, name)\n1001:             actions.append((func, name, description))\n1002:         # Add actions from this ModelAdmin.\n1003:         actions.extend(base_actions)\n1004:         return actions\n1005: \n1006:     def _filter_actions_by_permissions(self, request, actions):\n1007:         \"\"\"Filter out any actions that the user doesn't have access to.\"\"\"\n1008:         filtered_actions = []\n1009:         for action in actions:\n1010:             callable = action[0]\n1011:             if not hasattr(callable, \"allowed_permissions\"):\n1012:                 filtered_actions.append(action)\n1013:                 continue\n1014:             permission_checks = (\n1015:                 getattr(self, \"has_%s_permission\" % permission)\n1016:                 for permission in callable.allowed_permissions\n1017:             )\n1018:             if any(has_permission(request) for has_permission in permission_checks):\n1019:                 filtered_actions.append(action)\n1020:         return filtered_actions\n1021: \n1022:     def get_actions(self, request):\n1023:         \"\"\"\n1024:         Return a dictionary mapping the names of all actions for this\n1025:         ModelAdmin to a tuple of (callable, name, description) for each action.\n1026:         \"\"\"\n1027:         # If self.actions is set to None that means actions are disabled on\n1028:         # this page.\n1029:         if self.actions is None or IS_POPUP_VAR in request.GET:\n1030:             return {}\n1031:         actions = self._filter_actions_by_permissions(request, self._get_base_actions())\n1032:         return {name: (func, name, desc) for func, name, desc in actions}\n1033: \n1034:     def get_action_choices(self, request, default_choices=models.BLANK_CHOICE_DASH):\n1035:         \"\"\"\n1036:         Return a list of choices for use in a form object.  Each choice is a\n1037:         tuple (name, description).\n1038:         \"\"\"\n1039:         choices = [] + default_choices\n1040:         for func, name, description in self.get_actions(request).values():\n1041:             choice = (name, description % model_format_dict(self.opts))\n1042:             choices.append(choice)\n1043:         return choices\n1044: \n1045:     def get_action(self, action):\n1046:         \"\"\"\n1047:         Return a given action from a parameter, which can either be a callable,\n1048:         or the name of a method on the ModelAdmin.  Return is a tuple of\n1049:         (callable, name, description).\n1050:         \"\"\"\n1051:         # If the action is a callable, just use it.\n1052:         if callable(action):\n1053:             func = action\n1054:             action = action.__name__\n1055: \n1056:         # Next, look for a method. Grab it off self.__class__ to get an unbound\n1057:         # method instead of a bound one; this ensures that the calling\n1058:         # conventions are the same for functions and methods.\n1059:         elif hasattr(self.__class__, action):\n1060:             func = getattr(self.__class__, action)\n1061: \n1062:         # Finally, look for a named method on the admin site\n1063:         else:\n1064:             try:\n1065:                 func = self.admin_site.get_action(action)\n1066:             except KeyError:\n1067:                 return None\n1068: \n1069:         description = self._get_action_description(func, action)\n1070:         return func, action, description\n1071: \n1072:     def get_list_display(self, request):\n1073:         \"\"\"\n1074:         Return a sequence containing the fields to be displayed on the\n1075:         changelist.\n1076:         \"\"\"\n1077:         return self.list_display\n1078: \n1079:     def get_list_display_links(self, request, list_display):\n1080:         \"\"\"\n1081:         Return a sequence containing the fields to be displayed as links\n1082:         on the changelist. The list_display parameter is the list of fields\n1083:         returned by get_list_display().\n1084:         \"\"\"\n1085:         if (\n1086:             self.list_display_links\n1087:             or self.list_display_links is None\n1088:             or not list_display\n1089:         ):\n1090:             return self.list_display_links\n1091:         else:\n1092:             # Use only the first item in list_display as link\n1093:             return list(list_display)[:1]\n1094: \n1095:     def get_list_filter(self, request):\n1096:         \"\"\"\n1097:         Return a sequence containing the fields to be displayed as filters in\n1098:         the right sidebar of the changelist page.\n1099:         \"\"\"\n1100:         return self.list_filter\n1101: \n1102:     def get_list_select_related(self, request):\n1103:         \"\"\"\n1104:         Return a list of fields to add to the select_related() part of the\n1105:         changelist items query.\n1106:         \"\"\"\n1107:         return self.list_select_related\n1108: \n1109:     def get_search_fields(self, request):\n1110:         \"\"\"\n1111:         Return a sequence containing the fields to be searched whenever\n1112:         somebody submits a search query.\n1113:         \"\"\"\n1114:         return self.search_fields\n1115: \n1116:     def get_search_results(self, request, queryset, search_term):\n1117:         \"\"\"\n1118:         Return a tuple containing a queryset to implement the search\n1119:         and a boolean indicating if the results may contain duplicates.\n1120:         \"\"\"\n1121: \n1122:         # Apply keyword searches.\n1123:         def construct_search(field_name):\n1124:             if field_name.startswith(\"^\"):\n1125:                 return \"%s__istartswith\" % field_name.removeprefix(\"^\")\n1126:             elif field_name.startswith(\"=\"):\n1127:                 return \"%s__iexact\" % field_name.removeprefix(\"=\")\n1128:             elif field_name.startswith(\"@\"):\n1129:                 return \"%s__search\" % field_name.removeprefix(\"@\")\n1130:             # Use field_name if it includes a lookup.\n1131:             opts = queryset.model._meta\n1132:             lookup_fields = field_name.split(LOOKUP_SEP)\n1133:             # Go through the fields, following all relations.\n1134:             prev_field = None\n1135:             for path_part in lookup_fields:\n1136:                 if path_part == \"pk\":\n1137:                     path_part = opts.pk.name\n1138:                 try:\n1139:                     field = opts.get_field(path_part)\n1140:                 except FieldDoesNotExist:\n1141:                     # Use valid query lookups.\n1142:                     if prev_field and prev_field.get_lookup(path_part):\n1143:                         return field_name\n1144:                 else:\n1145:                     prev_field = field\n1146:                     if hasattr(field, \"path_infos\"):\n1147:                         # Update opts to follow the relation.\n1148:                         opts = field.path_infos[-1].to_opts\n1149:             # Otherwise, use the field with icontains.\n1150:             return \"%s__icontains\" % field_name\n1151: \n1152:         may_have_duplicates = False\n1153:         search_fields = self.get_search_fields(request)\n1154:         if search_fields and search_term:\n1155:             orm_lookups = [\n1156:                 construct_search(str(search_field)) for search_field in search_fields\n1157:             ]\n1158:             term_queries = []\n1159:             for bit in smart_split(search_term):\n1160:                 if bit.startswith(('\"', \"'\")) and bit[0] == bit[-1]:\n1161:                     bit = unescape_string_literal(bit)\n1162:                 or_queries = models.Q.create(\n1163:                     [(orm_lookup, bit) for orm_lookup in orm_lookups],\n1164:                     connector=models.Q.OR,\n1165:                 )\n1166:                 term_queries.append(or_queries)\n1167:             queryset = queryset.filter(models.Q.create(term_queries))\n1168:             may_have_duplicates |= any(\n1169:                 lookup_spawns_duplicates(self.opts, search_spec)\n1170:                 for search_spec in orm_lookups\n1171:             )\n1172:         return queryset, may_have_duplicates\n1173: \n1174:     def get_preserved_filters(self, request):\n1175:         \"\"\"\n1176:         Return the preserved filters querystring.\n1177:         \"\"\"\n1178:         match = request.resolver_match\n1179:         if self.preserve_filters and match:\n1180:             current_url = \"%s:%s\" % (match.app_name, match.url_name)\n1181:             changelist_url = \"admin:%s_%s_changelist\" % (\n1182:                 self.opts.app_label,\n1183:                 self.opts.model_name,\n1184:             )\n1185:             if current_url == changelist_url:\n1186:                 preserved_filters = request.GET.urlencode()\n1187:             else:\n1188:                 preserved_filters = request.GET.get(\"_changelist_filters\")\n1189: \n1190:             if preserved_filters:\n1191:                 return urlencode({\"_changelist_filters\": preserved_filters})\n1192:         return \"\"\n1193: \n1194:     def construct_change_message(self, request, form, formsets, add=False):\n1195:         \"\"\"\n1196:         Construct a JSON structure describing changes from a changed object.\n1197:         \"\"\"\n1198:         return construct_change_message(form, formsets, add)\n1199: \n1200:     def message_user(\n1201:         self, request, message, level=messages.INFO, extra_tags=\"\", fail_silently=False\n1202:     ):\n1203:         \"\"\"\n1204:         Send a message to the user. The default implementation\n1205:         posts a message using the django.contrib.messages backend.\n1206: \n1207:         Exposes almost the same API as messages.add_message(), but accepts the\n1208:         positional arguments in a different order to maintain backwards\n1209:         compatibility. For convenience, it accepts the `level` argument as\n1210:         a string rather than the usual level number.\n1211:         \"\"\"\n1212:         if not isinstance(level, int):\n1213:             # attempt to get the level if passed a string\n1214:             try:\n1215:                 level = getattr(messages.constants, level.upper())\n1216:             except AttributeError:\n1217:                 levels = messages.constants.DEFAULT_TAGS.values()\n1218:                 levels_repr = \", \".join(\"`%s`\" % level for level in levels)\n1219:                 raise ValueError(\n1220:                     \"Bad message level string: `%s`. Possible values are: %s\"\n1221:                     % (level, levels_repr)\n1222:                 )\n1223: \n1224:         messages.add_message(\n1225:             request, level, message, extra_tags=extra_tags, fail_silently=fail_silently\n1226:         )\n1227: \n1228:     def save_form(self, request, form, change):\n1229:         \"\"\"\n1230:         Given a ModelForm return an unsaved instance. ``change`` is True if\n1231:         the object is being changed, and False if it's being added.\n1232:         \"\"\"\n1233:         return form.save(commit=False)\n1234: \n1235:     def save_model(self, request, obj, form, change):\n1236:         \"\"\"\n1237:         Given a model instance save it to the database.\n1238:         \"\"\"\n1239:         obj.save()\n1240: \n1241:     def delete_model(self, request, obj):\n1242:         \"\"\"\n1243:         Given a model instance delete it from the database.\n1244:         \"\"\"\n1245:         obj.delete()\n1246: \n1247:     def delete_queryset(self, request, queryset):\n1248:         \"\"\"Given a queryset, delete it from the database.\"\"\"\n1249:         queryset.delete()\n1250: \n1251:     def save_formset(self, request, form, formset, change):\n1252:         \"\"\"\n1253:         Given an inline formset save it to the database.\n1254:         \"\"\"\n1255:         formset.save()\n1256: \n1257:     def save_related(self, request, form, formsets, change):\n1258:         \"\"\"\n1259:         Given the ``HttpRequest``, the parent ``ModelForm`` instance, the\n1260:         list of inline formsets and a boolean value based on whether the\n1261:         parent is being added or changed, save the related objects to the\n1262:         database. Note that at this point save_form() and save_model() have\n1263:         already been called.\n1264:         \"\"\"\n1265:         form.save_m2m()\n1266:         for formset in formsets:\n1267:             self.save_formset(request, form, formset, change=change)\n1268: \n1269:     def render_change_form(\n1270:         self, request, context, add=False, change=False, form_url=\"\", obj=None\n1271:     ):\n1272:         app_label = self.opts.app_label\n1273:         preserved_filters = self.get_preserved_filters(request)\n1274:         form_url = add_preserved_filters(\n1275:             {\"preserved_filters\": preserved_filters, \"opts\": self.opts}, form_url\n1276:         )\n1277:         view_on_site_url = self.get_view_on_site_url(obj)\n1278:         has_editable_inline_admin_formsets = False\n1279:         for inline in context[\"inline_admin_formsets\"]:\n1280:             if (\n1281:                 inline.has_add_permission\n1282:                 or inline.has_change_permission\n1283:                 or inline.has_delete_permission\n1284:             ):\n1285:                 has_editable_inline_admin_formsets = True\n1286:                 break\n1287:         context.update(\n1288:             {\n1289:                 \"add\": add,\n1290:                 \"change\": change,\n1291:                 \"has_view_permission\": self.has_view_permission(request, obj),\n1292:                 \"has_add_permission\": self.has_add_permission(request),\n1293:                 \"has_change_permission\": self.has_change_permission(request, obj),\n1294:                 \"has_delete_permission\": self.has_delete_permission(request, obj),\n1295:                 \"has_editable_inline_admin_formsets\": (\n1296:                     has_editable_inline_admin_formsets\n1297:                 ),\n1298:                 \"has_file_field\": context[\"adminform\"].form.is_multipart()\n1299:                 or any(\n1300:                     admin_formset.formset.is_multipart()\n1301:                     for admin_formset in context[\"inline_admin_formsets\"]\n1302:                 ),\n1303:                 \"has_absolute_url\": view_on_site_url is not None,\n1304:                 \"absolute_url\": view_on_site_url,\n1305:                 \"form_url\": form_url,\n1306:                 \"opts\": self.opts,\n1307:                 \"content_type_id\": get_content_type_for_model(self.model).pk,\n1308:                 \"save_as\": self.save_as,\n1309:                 \"save_on_top\": self.save_on_top,\n1310:                 \"to_field_var\": TO_FIELD_VAR,\n1311:                 \"is_popup_var\": IS_POPUP_VAR,\n1312:                 \"app_label\": app_label,\n1313:             }\n1314:         )\n1315:         if add and self.add_form_template is not None:\n1316:             form_template = self.add_form_template\n1317:         else:\n1318:             form_template = self.change_form_template\n1319: \n1320:         request.current_app = self.admin_site.name\n1321: \n1322:         return TemplateResponse(\n1323:             request,\n1324:             form_template\n1325:             or [\n1326:                 \"admin/%s/%s/change_form.html\" % (app_label, self.opts.model_name),\n1327:                 \"admin/%s/change_form.html\" % app_label,\n1328:                 \"admin/change_form.html\",\n1329:             ],\n1330:             context,\n1331:         )\n1332: \n1333:     def response_add(self, request, obj, post_url_continue=None):\n1334:         \"\"\"\n1335:         Determine the HttpResponse for the add_view stage.\n1336:         \"\"\"\n1337:         opts = obj._meta\n1338:         preserved_filters = self.get_preserved_filters(request)\n1339:         obj_url = reverse(\n1340:             \"admin:%s_%s_change\" % (opts.app_label, opts.model_name),\n1341:             args=(quote(obj.pk),),\n1342:             current_app=self.admin_site.name,\n1343:         )\n1344:         # Add a link to the object's change form if the user can edit the obj.\n1345:         if self.has_change_permission(request, obj):\n1346:             obj_repr = format_html('<a href=\"{}\">{}</a>', urlquote(obj_url), obj)\n1347:         else:\n1348:             obj_repr = str(obj)\n1349:         msg_dict = {\n1350:             \"name\": opts.verbose_name,\n1351:             \"obj\": obj_repr,\n1352:         }\n1353:         # Here, we distinguish between different save types by checking for\n1354:         # the presence of keys in request.POST.\n1355: \n1356:         if IS_POPUP_VAR in request.POST:\n1357:             to_field = request.POST.get(TO_FIELD_VAR)\n1358:             if to_field:\n1359:                 attr = str(to_field)\n1360:             else:\n1361:                 attr = obj._meta.pk.attname\n1362:             value = obj.serializable_value(attr)\n1363:             popup_response_data = json.dumps(\n1364:                 {\n1365:                     \"value\": str(value),\n1366:                     \"obj\": str(obj),\n1367:                 }\n1368:             )\n1369:             return TemplateResponse(\n1370:                 request,\n1371:                 self.popup_response_template\n1372:                 or [\n1373:                     \"admin/%s/%s/popup_response.html\"\n1374:                     % (opts.app_label, opts.model_name),\n1375:                     \"admin/%s/popup_response.html\" % opts.app_label,\n1376:                     \"admin/popup_response.html\",\n1377:                 ],\n1378:                 {\n1379:                     \"popup_response_data\": popup_response_data,\n1380:                 },\n1381:             )\n1382: \n1383:         elif \"_continue\" in request.POST or (\n1384:             # Redirecting after \"Save as new\".\n1385:             \"_saveasnew\" in request.POST\n1386:             and self.save_as_continue\n1387:             and self.has_change_permission(request, obj)\n1388:         ):\n1389:             msg = _(\"The {name} \u201c{obj}\u201d was added successfully.\")\n1390:             if self.has_change_permission(request, obj):\n1391:                 msg += \" \" + _(\"You may edit it again below.\")\n1392:             self.message_user(request, format_html(msg, **msg_dict), messages.SUCCESS)\n1393:             if post_url_continue is None:\n1394:                 post_url_continue = obj_url\n1395:             post_url_continue = add_preserved_filters(\n1396:                 {\"preserved_filters\": preserved_filters, \"opts\": opts},\n1397:                 post_url_continue,\n1398:             )\n1399:             return HttpResponseRedirect(post_url_continue)\n1400: \n1401:         elif \"_addanother\" in request.POST:\n1402:             msg = format_html(\n1403:                 _(\n1404:                     \"The {name} \u201c{obj}\u201d was added successfully. You may add another \"\n1405:                     \"{name} below.\"\n1406:                 ),\n1407:                 **msg_dict,\n1408:             )\n1409:             self.message_user(request, msg, messages.SUCCESS)\n1410:             redirect_url = request.path\n1411:             redirect_url = add_preserved_filters(\n1412:                 {\"preserved_filters\": preserved_filters, \"opts\": opts}, redirect_url\n1413:             )\n1414:             return HttpResponseRedirect(redirect_url)\n1415: \n1416:         else:\n1417:             msg = format_html(\n1418:                 _(\"The {name} \u201c{obj}\u201d was added successfully.\"), **msg_dict\n1419:             )\n1420:             self.message_user(request, msg, messages.SUCCESS)\n1421:             return self.response_post_save_add(request, obj)\n1422: \n1423:     def response_change(self, request, obj):\n1424:         \"\"\"\n1425:         Determine the HttpResponse for the change_view stage.\n1426:         \"\"\"\n1427: \n1428:         if IS_POPUP_VAR in request.POST:\n1429:             opts = obj._meta\n1430:             to_field = request.POST.get(TO_FIELD_VAR)\n1431:             attr = str(to_field) if to_field else opts.pk.attname\n1432:             value = request.resolver_match.kwargs[\"object_id\"]\n1433:             new_value = obj.serializable_value(attr)\n1434:             popup_response_data = json.dumps(\n1435:                 {\n1436:                     \"action\": \"change\",\n1437:                     \"value\": str(value),\n1438:                     \"obj\": str(obj),\n1439:                     \"new_value\": str(new_value),\n1440:                 }\n1441:             )\n1442:             return TemplateResponse(\n1443:                 request,\n1444:                 self.popup_response_template\n1445:                 or [\n1446:                     \"admin/%s/%s/popup_response.html\"\n1447:                     % (opts.app_label, opts.model_name),\n1448:                     \"admin/%s/popup_response.html\" % opts.app_label,\n1449:                     \"admin/popup_response.html\",\n1450:                 ],\n1451:                 {\n1452:                     \"popup_response_data\": popup_response_data,\n1453:                 },\n1454:             )\n1455: \n1456:         opts = self.opts\n1457:         preserved_filters = self.get_preserved_filters(request)\n1458: \n1459:         msg_dict = {\n1460:             \"name\": opts.verbose_name,\n1461:             \"obj\": format_html('<a href=\"{}\">{}</a>', urlquote(request.path), obj),\n1462:         }\n1463:         if \"_continue\" in request.POST:\n1464:             msg = format_html(\n1465:                 _(\n1466:                     \"The {name} \u201c{obj}\u201d was changed successfully. You may edit it \"\n1467:                     \"again below.\"\n1468:                 ),\n1469:                 **msg_dict,\n1470:             )\n1471:             self.message_user(request, msg, messages.SUCCESS)\n1472:             redirect_url = request.path\n1473:             redirect_url = add_preserved_filters(\n1474:                 {\"preserved_filters\": preserved_filters, \"opts\": opts}, redirect_url\n1475:             )\n1476:             return HttpResponseRedirect(redirect_url)\n1477: \n1478:         elif \"_saveasnew\" in request.POST:\n1479:             msg = format_html(\n1480:                 _(\n1481:                     \"The {name} \u201c{obj}\u201d was added successfully. You may edit it again \"\n1482:                     \"below.\"\n1483:                 ),\n1484:                 **msg_dict,\n1485:             )\n1486:             self.message_user(request, msg, messages.SUCCESS)\n1487:             redirect_url = reverse(\n1488:                 \"admin:%s_%s_change\" % (opts.app_label, opts.model_name),\n1489:                 args=(obj.pk,),\n1490:                 current_app=self.admin_site.name,\n1491:             )\n1492:             redirect_url = add_preserved_filters(\n1493:                 {\"preserved_filters\": preserved_filters, \"opts\": opts}, redirect_url\n1494:             )\n1495:             return HttpResponseRedirect(redirect_url)\n1496: \n1497:         elif \"_addanother\" in request.POST:\n1498:             msg = format_html(\n1499:                 _(\n1500:                     \"The {name} \u201c{obj}\u201d was changed successfully. You may add another \"\n1501:                     \"{name} below.\"\n1502:                 ),\n1503:                 **msg_dict,\n1504:             )\n1505:             self.message_user(request, msg, messages.SUCCESS)\n1506:             redirect_url = reverse(\n1507:                 \"admin:%s_%s_add\" % (opts.app_label, opts.model_name),\n1508:                 current_app=self.admin_site.name,\n1509:             )\n1510:             redirect_url = add_preserved_filters(\n1511:                 {\"preserved_filters\": preserved_filters, \"opts\": opts}, redirect_url\n1512:             )\n1513:             return HttpResponseRedirect(redirect_url)\n1514: \n1515:         else:\n1516:             msg = format_html(\n1517:                 _(\"The {name} \u201c{obj}\u201d was changed successfully.\"), **msg_dict\n1518:             )\n1519:             self.message_user(request, msg, messages.SUCCESS)\n1520:             return self.response_post_save_change(request, obj)\n1521: \n1522:     def _response_post_save(self, request, obj):\n1523:         if self.has_view_or_change_permission(request):\n1524:             post_url = reverse(\n1525:                 \"admin:%s_%s_changelist\" % (self.opts.app_label, self.opts.model_name),\n1526:                 current_app=self.admin_site.name,\n1527:             )\n1528:             preserved_filters = self.get_preserved_filters(request)\n1529:             post_url = add_preserved_filters(\n1530:                 {\"preserved_filters\": preserved_filters, \"opts\": self.opts}, post_url\n1531:             )\n1532:         else:\n1533:             post_url = reverse(\"admin:index\", current_app=self.admin_site.name)\n1534:         return HttpResponseRedirect(post_url)\n1535: \n1536:     def response_post_save_add(self, request, obj):\n1537:         \"\"\"\n1538:         Figure out where to redirect after the 'Save' button has been pressed\n1539:         when adding a new object.\n1540:         \"\"\"\n1541:         return self._response_post_save(request, obj)\n1542: \n1543:     def response_post_save_change(self, request, obj):\n1544:         \"\"\"\n1545:         Figure out where to redirect after the 'Save' button has been pressed\n1546:         when editing an existing object.\n1547:         \"\"\"\n1548:         return self._response_post_save(request, obj)\n1549: \n1550:     def response_action(self, request, queryset):\n1551:         \"\"\"\n1552:         Handle an admin action. This is called if a request is POSTed to the\n1553:         changelist; it returns an HttpResponse if the action was handled, and\n1554:         None otherwise.\n1555:         \"\"\"\n1556: \n1557:         # There can be multiple action forms on the page (at the top\n1558:         # and bottom of the change list, for example). Get the action\n1559:         # whose button was pushed.\n1560:         try:\n1561:             action_index = int(request.POST.get(\"index\", 0))\n1562:         except ValueError:\n1563:             action_index = 0\n1564: \n1565:         # Construct the action form.\n1566:         data = request.POST.copy()\n1567:         data.pop(helpers.ACTION_CHECKBOX_NAME, None)\n1568:         data.pop(\"index\", None)\n1569: \n1570:         # Use the action whose button was pushed\n1571:         try:\n1572:             data.update({\"action\": data.getlist(\"action\")[action_index]})\n1573:         except IndexError:\n1574:             # If we didn't get an action from the chosen form that's invalid\n1575:             # POST data, so by deleting action it'll fail the validation check\n1576:             # below. So no need to do anything here\n1577:             pass\n1578: \n1579:         action_form = self.action_form(data, auto_id=None)\n1580:         action_form.fields[\"action\"].choices = self.get_action_choices(request)\n1581: \n1582:         # If the form's valid we can handle the action.\n1583:         if action_form.is_valid():\n1584:             action = action_form.cleaned_data[\"action\"]\n1585:             select_across = action_form.cleaned_data[\"select_across\"]\n1586:             func = self.get_actions(request)[action][0]\n1587: \n1588:             # Get the list of selected PKs. If nothing's selected, we can't\n1589:             # perform an action on it, so bail. Except we want to perform\n1590:             # the action explicitly on all objects.\n1591:             selected = request.POST.getlist(helpers.ACTION_CHECKBOX_NAME)\n1592:             if not selected and not select_across:\n1593:                 # Reminder that something needs to be selected or nothing will happen\n1594:                 msg = _(\n1595:                     \"Items must be selected in order to perform \"\n1596:                     \"actions on them. No items have been changed.\"\n1597:                 )\n1598:                 self.message_user(request, msg, messages.WARNING)\n1599:                 return None\n1600: \n1601:             if not select_across:\n1602:                 # Perform the action only on the selected objects\n1603:                 queryset = queryset.filter(pk__in=selected)\n1604: \n1605:             response = func(self, request, queryset)\n1606: \n1607:             # Actions may return an HttpResponse-like object, which will be\n1608:             # used as the response from the POST. If not, we'll be a good\n1609:             # little HTTP citizen and redirect back to the changelist page.\n1610:             if isinstance(response, HttpResponseBase):\n1611:                 return response\n1612:             else:\n1613:                 return HttpResponseRedirect(request.get_full_path())\n1614:         else:\n1615:             msg = _(\"No action selected.\")\n1616:             self.message_user(request, msg, messages.WARNING)\n1617:             return None\n1618: \n1619:     def response_delete(self, request, obj_display, obj_id):\n1620:         \"\"\"\n1621:         Determine the HttpResponse for the delete_view stage.\n1622:         \"\"\"\n1623:         if IS_POPUP_VAR in request.POST:\n1624:             popup_response_data = json.dumps(\n1625:                 {\n1626:                     \"action\": \"delete\",\n1627:                     \"value\": str(obj_id),\n1628:                 }\n1629:             )\n1630:             return TemplateResponse(\n1631:                 request,\n1632:                 self.popup_response_template\n1633:                 or [\n1634:                     \"admin/%s/%s/popup_response.html\"\n1635:                     % (self.opts.app_label, self.opts.model_name),\n1636:                     \"admin/%s/popup_response.html\" % self.opts.app_label,\n1637:                     \"admin/popup_response.html\",\n1638:                 ],\n1639:                 {\n1640:                     \"popup_response_data\": popup_response_data,\n1641:                 },\n1642:             )\n1643: \n1644:         self.message_user(\n1645:             request,\n1646:             _(\"The %(name)s \u201c%(obj)s\u201d was deleted successfully.\")\n1647:             % {\n1648:                 \"name\": self.opts.verbose_name,\n1649:                 \"obj\": obj_display,\n1650:             },\n1651:             messages.SUCCESS,\n1652:         )\n1653: \n1654:         if self.has_change_permission(request, None):\n1655:             post_url = reverse(\n1656:                 \"admin:%s_%s_changelist\" % (self.opts.app_label, self.opts.model_name),\n1657:                 current_app=self.admin_site.name,\n1658:             )\n1659:             preserved_filters = self.get_preserved_filters(request)\n1660:             post_url = add_preserved_filters(\n1661:                 {\"preserved_filters\": preserved_filters, \"opts\": self.opts}, post_url\n1662:             )\n1663:         else:\n1664:             post_url = reverse(\"admin:index\", current_app=self.admin_site.name)\n1665:         return HttpResponseRedirect(post_url)\n1666: \n1667:     def render_delete_form(self, request, context):\n1668:         app_label = self.opts.app_label\n1669: \n1670:         request.current_app = self.admin_site.name\n1671:         context.update(\n1672:             to_field_var=TO_FIELD_VAR,\n1673:             is_popup_var=IS_POPUP_VAR,\n1674:             media=self.media,\n1675:         )\n1676: \n1677:         return TemplateResponse(\n1678:             request,\n1679:             self.delete_confirmation_template\n1680:             or [\n1681:                 \"admin/{}/{}/delete_confirmation.html\".format(\n1682:                     app_label, self.opts.model_name\n1683:                 ),\n1684:                 \"admin/{}/delete_confirmation.html\".format(app_label),\n1685:                 \"admin/delete_confirmation.html\",\n1686:             ],\n1687:             context,\n1688:         )\n1689: \n1690:     def get_inline_formsets(self, request, formsets, inline_instances, obj=None):\n1691:         # Edit permissions on parent model are required for editable inlines.\n1692:         can_edit_parent = (\n1693:             self.has_change_permission(request, obj)\n1694:             if obj\n1695:             else self.has_add_permission(request)\n1696:         )\n1697:         inline_admin_formsets = []\n1698:         for inline, formset in zip(inline_instances, formsets):\n1699:             fieldsets = list(inline.get_fieldsets(request, obj))\n1700:             readonly = list(inline.get_readonly_fields(request, obj))\n1701:             if can_edit_parent:\n1702:                 has_add_permission = inline.has_add_permission(request, obj)\n1703:                 has_change_permission = inline.has_change_permission(request, obj)\n1704:                 has_delete_permission = inline.has_delete_permission(request, obj)\n1705:             else:\n1706:                 # Disable all edit-permissions, and override formset settings.\n1707:                 has_add_permission = (\n1708:                     has_change_permission\n1709:                 ) = has_delete_permission = False\n1710:                 formset.extra = formset.max_num = 0\n1711:             has_view_permission = inline.has_view_permission(request, obj)\n1712:             prepopulated = dict(inline.get_prepopulated_fields(request, obj))\n1713:             inline_admin_formset = helpers.InlineAdminFormSet(\n1714:                 inline,\n1715:                 formset,\n1716:                 fieldsets,\n1717:                 prepopulated,\n1718:                 readonly,\n1719:                 model_admin=self,\n1720:                 has_add_permission=has_add_permission,\n1721:                 has_change_permission=has_change_permission,\n1722:                 has_delete_permission=has_delete_permission,\n1723:                 has_view_permission=has_view_permission,\n1724:             )\n1725:             inline_admin_formsets.append(inline_admin_formset)\n1726:         return inline_admin_formsets\n1727: \n1728:     def get_changeform_initial_data(self, request):\n1729:         \"\"\"\n1730:         Get the initial form data from the request's GET params.\n1731:         \"\"\"\n1732:         initial = dict(request.GET.items())\n1733:         for k in initial:\n1734:             try:\n1735:                 f = self.opts.get_field(k)\n1736:             except FieldDoesNotExist:\n1737:                 continue\n1738:             # We have to special-case M2Ms as a list of comma-separated PKs.\n1739:             if isinstance(f, models.ManyToManyField):\n1740:                 initial[k] = initial[k].split(\",\")\n1741:         return initial\n1742: \n1743:     def _get_obj_does_not_exist_redirect(self, request, opts, object_id):\n1744:         \"\"\"\n1745:         Create a message informing the user that the object doesn't exist\n1746:         and return a redirect to the admin index page.\n1747:         \"\"\"\n1748:         msg = _(\"%(name)s with ID \u201c%(key)s\u201d doesn\u2019t exist. Perhaps it was deleted?\") % {\n1749:             \"name\": opts.verbose_name,\n1750:             \"key\": unquote(object_id),\n1751:         }\n1752:         self.message_user(request, msg, messages.WARNING)\n1753:         url = reverse(\"admin:index\", current_app=self.admin_site.name)\n1754:         return HttpResponseRedirect(url)\n1755: \n1756:     @csrf_protect_m\n1757:     def changeform_view(self, request, object_id=None, form_url=\"\", extra_context=None):\n1758:         with transaction.atomic(using=router.db_for_write(self.model)):\n1759:             return self._changeform_view(request, object_id, form_url, extra_context)\n1760: \n1761:     def _changeform_view(self, request, object_id, form_url, extra_context):\n1762:         to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))\n1763:         if to_field and not self.to_field_allowed(request, to_field):\n1764:             raise DisallowedModelAdminToField(\n1765:                 \"The field %s cannot be referenced.\" % to_field\n1766:             )\n1767: \n1768:         if request.method == \"POST\" and \"_saveasnew\" in request.POST:\n1769:             object_id = None\n1770: \n1771:         add = object_id is None\n1772: \n1773:         if add:\n1774:             if not self.has_add_permission(request):\n1775:                 raise PermissionDenied\n1776:             obj = None\n1777: \n1778:         else:\n1779:             obj = self.get_object(request, unquote(object_id), to_field)\n1780: \n1781:             if request.method == \"POST\":\n1782:                 if not self.has_change_permission(request, obj):\n1783:                     raise PermissionDenied\n1784:             else:\n1785:                 if not self.has_view_or_change_permission(request, obj):\n1786:                     raise PermissionDenied\n1787: \n1788:             if obj is None:\n1789:                 return self._get_obj_does_not_exist_redirect(\n1790:                     request, self.opts, object_id\n1791:                 )\n1792: \n1793:         fieldsets = self.get_fieldsets(request, obj)\n1794:         ModelForm = self.get_form(\n1795:             request, obj, change=not add, fields=flatten_fieldsets(fieldsets)\n1796:         )\n1797:         if request.method == \"POST\":\n1798:             form = ModelForm(request.POST, request.FILES, instance=obj)\n1799:             formsets, inline_instances = self._create_formsets(\n1800:                 request,\n1801:                 form.instance,\n1802:                 change=not add,\n1803:             )\n1804:             form_validated = form.is_valid()\n1805:             if form_validated:\n1806:                 new_object = self.save_form(request, form, change=not add)\n1807:             else:\n1808:                 new_object = form.instance\n1809:             if all_valid(formsets) and form_validated:\n1810:                 self.save_model(request, new_object, form, not add)\n1811:                 self.save_related(request, form, formsets, not add)\n1812:                 change_message = self.construct_change_message(\n1813:                     request, form, formsets, add\n1814:                 )\n1815:                 if add:\n1816:                     self.log_addition(request, new_object, change_message)\n1817:                     return self.response_add(request, new_object)\n1818:                 else:\n1819:                     self.log_change(request, new_object, change_message)\n1820:                     return self.response_change(request, new_object)\n1821:             else:\n1822:                 form_validated = False\n1823:         else:\n1824:             if add:\n1825:                 initial = self.get_changeform_initial_data(request)\n1826:                 form = ModelForm(initial=initial)\n1827:                 formsets, inline_instances = self._create_formsets(\n1828:                     request, form.instance, change=False\n1829:                 )\n1830:             else:\n1831:                 form = ModelForm(instance=obj)\n1832:                 formsets, inline_instances = self._create_formsets(\n1833:                     request, obj, change=True\n1834:                 )\n1835: \n1836:         if not add and not self.has_change_permission(request, obj):\n1837:             readonly_fields = flatten_fieldsets(fieldsets)\n1838:         else:\n1839:             readonly_fields = self.get_readonly_fields(request, obj)\n1840:         admin_form = helpers.AdminForm(\n1841:             form,\n1842:             list(fieldsets),\n1843:             # Clear prepopulated fields on a view-only form to avoid a crash.\n1844:             self.get_prepopulated_fields(request, obj)\n1845:             if add or self.has_change_permission(request, obj)\n1846:             else {},\n1847:             readonly_fields,\n1848:             model_admin=self,\n1849:         )\n1850:         media = self.media + admin_form.media\n1851: \n1852:         inline_formsets = self.get_inline_formsets(\n1853:             request, formsets, inline_instances, obj\n1854:         )\n1855:         for inline_formset in inline_formsets:\n1856:             media += inline_formset.media\n1857: \n1858:         if add:\n1859:             title = _(\"Add %s\")\n1860:         elif self.has_change_permission(request, obj):\n1861:             title = _(\"Change %s\")\n1862:         else:\n1863:             title = _(\"View %s\")\n1864:         context = {\n1865:             **self.admin_site.each_context(request),\n1866:             \"title\": title % self.opts.verbose_name,\n1867:             \"subtitle\": str(obj) if obj else None,\n1868:             \"adminform\": admin_form,\n1869:             \"object_id\": object_id,\n1870:             \"original\": obj,\n1871:             \"is_popup\": IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET,\n1872:             \"to_field\": to_field,\n1873:             \"media\": media,\n1874:             \"inline_admin_formsets\": inline_formsets,\n1875:             \"errors\": helpers.AdminErrorList(form, formsets),\n1876:             \"preserved_filters\": self.get_preserved_filters(request),\n1877:         }\n1878: \n1879:         # Hide the \"Save\" and \"Save and continue\" buttons if \"Save as New\" was\n1880:         # previously chosen to prevent the interface from getting confusing.\n1881:         if (\n1882:             request.method == \"POST\"\n1883:             and not form_validated\n1884:             and \"_saveasnew\" in request.POST\n1885:         ):\n1886:             context[\"show_save\"] = False\n1887:             context[\"show_save_and_continue\"] = False\n1888:             # Use the change template instead of the add template.\n1889:             add = False\n1890: \n1891:         context.update(extra_context or {})\n1892: \n1893:         return self.render_change_form(\n1894:             request, context, add=add, change=not add, obj=obj, form_url=form_url\n1895:         )\n1896: \n1897:     def add_view(self, request, form_url=\"\", extra_context=None):\n1898:         return self.changeform_view(request, None, form_url, extra_context)\n1899: \n1900:     def change_view(self, request, object_id, form_url=\"\", extra_context=None):\n1901:         return self.changeform_view(request, object_id, form_url, extra_context)\n1902: \n1903:     def _get_edited_object_pks(self, request, prefix):\n1904:         \"\"\"Return POST data values of list_editable primary keys.\"\"\"\n1905:         pk_pattern = re.compile(\n1906:             r\"{}-\\d+-{}$\".format(re.escape(prefix), self.opts.pk.name)\n1907:         )\n1908:         return [value for key, value in request.POST.items() if pk_pattern.match(key)]\n1909: \n1910:     def _get_list_editable_queryset(self, request, prefix):\n1911:         \"\"\"\n1912:         Based on POST data, return a queryset of the objects that were edited\n1913:         via list_editable.\n1914:         \"\"\"\n1915:         object_pks = self._get_edited_object_pks(request, prefix)\n1916:         queryset = self.get_queryset(request)\n1917:         validate = queryset.model._meta.pk.to_python\n1918:         try:\n1919:             for pk in object_pks:\n1920:                 validate(pk)\n1921:         except ValidationError:\n1922:             # Disable the optimization if the POST data was tampered with.\n1923:             return queryset\n1924:         return queryset.filter(pk__in=object_pks)\n1925: \n1926:     @csrf_protect_m\n1927:     def changelist_view(self, request, extra_context=None):\n1928:         \"\"\"\n1929:         The 'change list' admin view for this model.\n1930:         \"\"\"\n1931:         from django.contrib.admin.views.main import ERROR_FLAG\n1932: \n1933:         app_label = self.opts.app_label\n1934:         if not self.has_view_or_change_permission(request):\n1935:             raise PermissionDenied\n1936: \n1937:         try:\n1938:             cl = self.get_changelist_instance(request)\n1939:         except IncorrectLookupParameters:\n1940:             # Wacky lookup parameters were given, so redirect to the main\n1941:             # changelist page, without parameters, and pass an 'invalid=1'\n1942:             # parameter via the query string. If wacky parameters were given\n1943:             # and the 'invalid=1' parameter was already in the query string,\n1944:             # something is screwed up with the database, so display an error\n1945:             # page.\n1946:             if ERROR_FLAG in request.GET:\n1947:                 return SimpleTemplateResponse(\n1948:                     \"admin/invalid_setup.html\",\n1949:                     {\n1950:                         \"title\": _(\"Database error\"),\n1951:                     },\n1952:                 )\n1953:             return HttpResponseRedirect(request.path + \"?\" + ERROR_FLAG + \"=1\")\n1954: \n1955:         # If the request was POSTed, this might be a bulk action or a bulk\n1956:         # edit. Try to look up an action or confirmation first, but if this\n1957:         # isn't an action the POST will fall through to the bulk edit check,\n1958:         # below.\n1959:         action_failed = False\n1960:         selected = request.POST.getlist(helpers.ACTION_CHECKBOX_NAME)\n1961: \n1962:         actions = self.get_actions(request)\n1963:         # Actions with no confirmation\n1964:         if (\n1965:             actions\n1966:             and request.method == \"POST\"\n1967:             and \"index\" in request.POST\n1968:             and \"_save\" not in request.POST\n1969:         ):\n1970:             if selected:\n1971:                 response = self.response_action(\n1972:                     request, queryset=cl.get_queryset(request)\n1973:                 )\n1974:                 if response:\n1975:                     return response\n1976:                 else:\n1977:                     action_failed = True\n1978:             else:\n1979:                 msg = _(\n1980:                     \"Items must be selected in order to perform \"\n1981:                     \"actions on them. No items have been changed.\"\n1982:                 )\n1983:                 self.message_user(request, msg, messages.WARNING)\n1984:                 action_failed = True\n1985: \n1986:         # Actions with confirmation\n1987:         if (\n1988:             actions\n1989:             and request.method == \"POST\"\n1990:             and helpers.ACTION_CHECKBOX_NAME in request.POST\n1991:             and \"index\" not in request.POST\n1992:             and \"_save\" not in request.POST\n1993:         ):\n1994:             if selected:\n1995:                 response = self.response_action(\n1996:                     request, queryset=cl.get_queryset(request)\n1997:                 )\n1998:                 if response:\n1999:                     return response\n2000:                 else:\n2001:                     action_failed = True\n2002: \n2003:         if action_failed:\n2004:             # Redirect back to the changelist page to avoid resubmitting the\n2005:             # form if the user refreshes the browser or uses the \"No, take\n2006:             # me back\" button on the action confirmation page.\n2007:             return HttpResponseRedirect(request.get_full_path())\n2008: \n2009:         # If we're allowing changelist editing, we need to construct a formset\n2010:         # for the changelist given all the fields to be edited. Then we'll\n2011:         # use the formset to validate/process POSTed data.\n2012:         formset = cl.formset = None\n2013: \n2014:         # Handle POSTed bulk-edit data.\n2015:         if request.method == \"POST\" and cl.list_editable and \"_save\" in request.POST:\n2016:             if not self.has_change_permission(request):\n2017:                 raise PermissionDenied\n2018:             FormSet = self.get_changelist_formset(request)\n2019:             modified_objects = self._get_list_editable_queryset(\n2020:                 request, FormSet.get_default_prefix()\n2021:             )\n2022:             formset = cl.formset = FormSet(\n2023:                 request.POST, request.FILES, queryset=modified_objects\n2024:             )\n2025:             if formset.is_valid():\n2026:                 changecount = 0\n2027:                 with transaction.atomic(using=router.db_for_write(self.model)):\n2028:                     for form in formset.forms:\n2029:                         if form.has_changed():\n2030:                             obj = self.save_form(request, form, change=True)\n2031:                             self.save_model(request, obj, form, change=True)\n2032:                             self.save_related(request, form, formsets=[], change=True)\n2033:                             change_msg = self.construct_change_message(\n2034:                                 request, form, None\n2035:                             )\n2036:                             self.log_change(request, obj, change_msg)\n2037:                             changecount += 1\n2038:                 if changecount:\n2039:                     msg = ngettext(\n2040:                         \"%(count)s %(name)s was changed successfully.\",\n2041:                         \"%(count)s %(name)s were changed successfully.\",\n2042:                         changecount,\n2043:                     ) % {\n2044:                         \"count\": changecount,\n2045:                         \"name\": model_ngettext(self.opts, changecount),\n2046:                     }\n2047:                     self.message_user(request, msg, messages.SUCCESS)\n2048: \n2049:                 return HttpResponseRedirect(request.get_full_path())\n2050: \n2051:         # Handle GET -- construct a formset for display.\n2052:         elif cl.list_editable and self.has_change_permission(request):\n2053:             FormSet = self.get_changelist_formset(request)\n2054:             formset = cl.formset = FormSet(queryset=cl.result_list)\n2055: \n2056:         # Build the list of media to be used by the formset.\n2057:         if formset:\n2058:             media = self.media + formset.media\n2059:         else:\n2060:             media = self.media\n2061: \n2062:         # Build the action form and populate it with available actions.\n2063:         if actions:\n2064:             action_form = self.action_form(auto_id=None)\n2065:             action_form.fields[\"action\"].choices = self.get_action_choices(request)\n2066:             media += action_form.media\n2067:         else:\n2068:             action_form = None\n2069: \n2070:         selection_note_all = ngettext(\n2071:             \"%(total_count)s selected\", \"All %(total_count)s selected\", cl.result_count\n2072:         )\n2073: \n2074:         context = {\n2075:             **self.admin_site.each_context(request),\n2076:             \"module_name\": str(self.opts.verbose_name_plural),\n2077:             \"selection_note\": _(\"0 of %(cnt)s selected\") % {\"cnt\": len(cl.result_list)},\n2078:             \"selection_note_all\": selection_note_all % {\"total_count\": cl.result_count},\n2079:             \"title\": cl.title,\n2080:             \"subtitle\": None,\n2081:             \"is_popup\": cl.is_popup,\n2082:             \"to_field\": cl.to_field,\n2083:             \"cl\": cl,\n2084:             \"media\": media,\n2085:             \"has_add_permission\": self.has_add_permission(request),\n2086:             \"opts\": cl.opts,\n2087:             \"action_form\": action_form,\n2088:             \"actions_on_top\": self.actions_on_top,\n2089:             \"actions_on_bottom\": self.actions_on_bottom,\n2090:             \"actions_selection_counter\": self.actions_selection_counter,\n2091:             \"preserved_filters\": self.get_preserved_filters(request),\n2092:             **(extra_context or {}),\n2093:         }\n2094: \n2095:         request.current_app = self.admin_site.name\n2096: \n2097:         return TemplateResponse(\n2098:             request,\n2099:             self.change_list_template\n2100:             or [\n2101:                 \"admin/%s/%s/change_list.html\" % (app_label, self.opts.model_name),\n2102:                 \"admin/%s/change_list.html\" % app_label,\n2103:                 \"admin/change_list.html\",\n2104:             ],\n2105:             context,\n2106:         )\n2107: \n2108:     def get_deleted_objects(self, objs, request):\n2109:         \"\"\"\n2110:         Hook for customizing the delete process for the delete view and the\n2111:         \"delete selected\" action.\n2112:         \"\"\"\n2113:         return get_deleted_objects(objs, request, self.admin_site)\n2114: \n2115:     @csrf_protect_m\n2116:     def delete_view(self, request, object_id, extra_context=None):\n2117:         with transaction.atomic(using=router.db_for_write(self.model)):\n2118:             return self._delete_view(request, object_id, extra_context)\n2119: \n2120:     def _delete_view(self, request, object_id, extra_context):\n2121:         \"The 'delete' admin view for this model.\"\n2122:         app_label = self.opts.app_label\n2123: \n2124:         to_field = request.POST.get(TO_FIELD_VAR, request.GET.get(TO_FIELD_VAR))\n2125:         if to_field and not self.to_field_allowed(request, to_field):\n2126:             raise DisallowedModelAdminToField(\n2127:                 \"The field %s cannot be referenced.\" % to_field\n2128:             )\n2129: \n2130:         obj = self.get_object(request, unquote(object_id), to_field)\n2131: \n2132:         if not self.has_delete_permission(request, obj):\n2133:             raise PermissionDenied\n2134: \n2135:         if obj is None:\n2136:             return self._get_obj_does_not_exist_redirect(request, self.opts, object_id)\n2137: \n2138:         # Populate deleted_objects, a data structure of all related objects that\n2139:         # will also be deleted.\n2140:         (\n2141:             deleted_objects,\n2142:             model_count,\n2143:             perms_needed,\n2144:             protected,\n2145:         ) = self.get_deleted_objects([obj], request)\n2146: \n2147:         if request.POST and not protected:  # The user has confirmed the deletion.\n2148:             if perms_needed:\n2149:                 raise PermissionDenied\n2150:             obj_display = str(obj)\n2151:             attr = str(to_field) if to_field else self.opts.pk.attname\n2152:             obj_id = obj.serializable_value(attr)\n2153:             self.log_deletion(request, obj, obj_display)\n2154:             self.delete_model(request, obj)\n2155: \n2156:             return self.response_delete(request, obj_display, obj_id)\n2157: \n2158:         object_name = str(self.opts.verbose_name)\n2159: \n2160:         if perms_needed or protected:\n2161:             title = _(\"Cannot delete %(name)s\") % {\"name\": object_name}\n2162:         else:\n2163:             title = _(\"Are you sure?\")\n2164: \n2165:         context = {\n2166:             **self.admin_site.each_context(request),\n2167:             \"title\": title,\n2168:             \"subtitle\": None,\n2169:             \"object_name\": object_name,\n2170:             \"object\": obj,\n2171:             \"deleted_objects\": deleted_objects,\n2172:             \"model_count\": dict(model_count).items(),\n2173:             \"perms_lacking\": perms_needed,\n2174:             \"protected\": protected,\n2175:             \"opts\": self.opts,\n2176:             \"app_label\": app_label,\n2177:             \"preserved_filters\": self.get_preserved_filters(request),\n2178:             \"is_popup\": IS_POPUP_VAR in request.POST or IS_POPUP_VAR in request.GET,\n2179:             \"to_field\": to_field,\n2180:             **(extra_context or {}),\n2181:         }\n2182: \n2183:         return self.render_delete_form(request, context)\n2184: \n2185:     def history_view(self, request, object_id, extra_context=None):\n2186:         \"The 'history' admin view for this model.\"\n2187:         from django.contrib.admin.models import LogEntry\n2188:         from django.contrib.admin.views.main import PAGE_VAR\n2189: \n2190:         # First check if the user can see this history.\n2191:         model = self.model\n2192:         obj = self.get_object(request, unquote(object_id))\n2193:         if obj is None:\n2194:             return self._get_obj_does_not_exist_redirect(\n2195:                 request, model._meta, object_id\n2196:             )\n2197: \n2198:         if not self.has_view_or_change_permission(request, obj):\n2199:             raise PermissionDenied\n2200: \n2201:         # Then get the history for this object.\n2202:         app_label = self.opts.app_label\n2203:         action_list = (\n2204:             LogEntry.objects.filter(\n2205:                 object_id=unquote(object_id),\n2206:                 content_type=get_content_type_for_model(model),\n2207:             )\n2208:             .select_related()\n2209:             .order_by(\"action_time\")\n2210:         )\n2211: \n2212:         paginator = self.get_paginator(request, action_list, 100)\n2213:         page_number = request.GET.get(PAGE_VAR, 1)\n2214:         page_obj = paginator.get_page(page_number)\n2215:         page_range = paginator.get_elided_page_range(page_obj.number)\n2216: \n2217:         context = {\n2218:             **self.admin_site.each_context(request),\n2219:             \"title\": _(\"Change history: %s\") % obj,\n2220:             \"subtitle\": None,\n2221:             \"action_list\": page_obj,\n2222:             \"page_range\": page_range,\n2223:             \"page_var\": PAGE_VAR,\n2224:             \"pagination_required\": paginator.count > 100,\n2225:             \"module_name\": str(capfirst(self.opts.verbose_name_plural)),\n2226:             \"object\": obj,\n2227:             \"opts\": self.opts,\n2228:             \"preserved_filters\": self.get_preserved_filters(request),\n2229:             **(extra_context or {}),\n2230:         }\n2231: \n2232:         request.current_app = self.admin_site.name\n2233: \n2234:         return TemplateResponse(\n2235:             request,\n2236:             self.object_history_template\n2237:             or [\n2238:                 \"admin/%s/%s/object_history.html\" % (app_label, self.opts.model_name),\n2239:                 \"admin/%s/object_history.html\" % app_label,\n2240:                 \"admin/object_history.html\",\n2241:             ],\n2242:             context,\n2243:         )\n2244: \n2245:     def get_formset_kwargs(self, request, obj, inline, prefix):\n2246:         formset_params = {\n2247:             \"instance\": obj,\n2248:             \"prefix\": prefix,\n2249:             \"queryset\": inline.get_queryset(request),\n2250:         }\n2251:         if request.method == \"POST\":\n2252:             formset_params.update(\n2253:                 {\n2254:                     \"data\": request.POST.copy(),\n2255:                     \"files\": request.FILES,\n2256:                     \"save_as_new\": \"_saveasnew\" in request.POST,\n2257:                 }\n2258:             )\n2259:         return formset_params\n2260: \n2261:     def _create_formsets(self, request, obj, change):\n2262:         \"Helper function to generate formsets for add/change_view.\"\n2263:         formsets = []\n2264:         inline_instances = []\n2265:         prefixes = {}\n2266:         get_formsets_args = [request]\n2267:         if change:\n2268:             get_formsets_args.append(obj)\n2269:         for FormSet, inline in self.get_formsets_with_inlines(*get_formsets_args):\n2270:             prefix = FormSet.get_default_prefix()\n2271:             prefixes[prefix] = prefixes.get(prefix, 0) + 1\n2272:             if prefixes[prefix] != 1 or not prefix:\n2273:                 prefix = \"%s-%s\" % (prefix, prefixes[prefix])\n2274:             formset_params = self.get_formset_kwargs(request, obj, inline, prefix)\n2275:             formset = FormSet(**formset_params)\n2276: \n2277:             def user_deleted_form(request, obj, formset, index, inline):\n2278:                 \"\"\"Return whether or not the user deleted the form.\"\"\"\n2279:                 return (\n2280:                     inline.has_delete_permission(request, obj)\n2281:                     and \"{}-{}-DELETE\".format(formset.prefix, index) in request.POST\n2282:                 )\n2283: \n2284:             # Bypass validation of each view-only inline form (since the form's\n2285:             # data won't be in request.POST), unless the form was deleted.\n2286:             if not inline.has_change_permission(request, obj if change else None):\n2287:                 for index, form in enumerate(formset.initial_forms):\n2288:                     if user_deleted_form(request, obj, formset, index, inline):\n2289:                         continue\n2290:                     form._errors = {}\n2291:                     form.cleaned_data = form.initial\n2292:             formsets.append(formset)\n2293:             inline_instances.append(inline)\n2294:         return formsets, inline_instances",
    "dispatch_relationship_context": [
      {
        "call": "format_html(_('The {name} \u201c{obj}\u201d was changed successfully. You may edit it again below.'), **msg_dict)",
        "call_name": "format_html",
        "consumer_candidates": [],
        "dispatch_key_candidates": [
          {
            "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
            "key": "unknown",
            "status": "unverified_dispatch_key",
            "values": []
          }
        ],
        "dispatcher_status": "wrapper_forwards_kwargs",
        "downranked_noise": [],
        "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
        "relation": "dispatcher_context",
        "source_node": {
          "id": "class:django/contrib/admin/options.py:ModelAdmin:621",
          "kind": "class",
          "lines": [
            621,
            2294
          ],
          "name": "ModelAdmin",
          "path": "django/contrib/admin/options.py"
        }
      },
      {
        "call": "format_html(_('The {name} \u201c{obj}\u201d was added successfully. You may edit it again below.'), **msg_dict)",
        "call_name": "format_html",
        "consumer_candidates": [],
        "dispatch_key_candidates": [
          {
            "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
            "key": "unknown",
            "status": "unverified_dispatch_key",
            "values": []
          }
        ],
        "dispatcher_status": "wrapper_forwards_kwargs",
        "downranked_noise": [],
        "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
        "relation": "dispatcher_context",
        "source_node": {
          "id": "class:django/contrib/admin/options.py:ModelAdmin:621",
          "kind": "class",
          "lines": [
            621,
            2294
          ],
          "name": "ModelAdmin",
          "path": "django/contrib/admin/options.py"
        }
      },
      {
        "call": "lookup_spawns_duplicates(self.opts, search_spec)",
        "call_name": "lookup_spawns_duplicates",
        "consumer_candidates": [],
        "dispatch_key_candidates": [
          {
            "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
            "key": "unknown",
            "status": "unverified_dispatch_key",
            "values": []
          }
        ],
        "dispatcher_status": "dispatcher_or_consumer_call",
        "downranked_noise": [],
        "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
        "relation": "dispatcher_context",
        "source_node": {
          "id": "class:django/contrib/admin/options.py:ModelAdmin:621",
          "kind": "class",
          "lines": [
            621,
            2294
          ],
          "name": "ModelAdmin",
          "path": "django/contrib/admin/options.py"
        }
      },
      {
        "call": "format_html(msg, **msg_dict)",
        "call_name": "format_html",
        "consumer_candidates": [],
        "dispatch_key_candidates": [
          {
            "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
            "key": "unknown",
            "status": "unverified_dispatch_key",
            "values": []
          }
        ],
        "dispatcher_status": "wrapper_forwards_kwargs",
        "downranked_noise": [],
        "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
        "relation": "dispatcher_context",
        "source_node": {
          "id": "class:django/contrib/admin/options.py:ModelAdmin:621",
          "kind": "class",
          "lines": [
            621,
            2294
          ],
          "name": "ModelAdmin",
          "path": "django/contrib/admin/options.py"
        }
      }
    ],
    "dispatch_tables": [],
    "local_symbol_references": [
      {
        "appearance_index": 17,
        "id": "class:django/contrib/admin/options.py:BaseModelAdmin:124",
        "kind": "class",
        "lines": [
          124,
          618
        ],
        "name": "BaseModelAdmin",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "BaseModelAdmin"
      },
      {
        "appearance_index": 521,
        "id": "class:django/contrib/admin/options.py:ShowFacets:75",
        "kind": "class",
        "lines": [
          75,
          78
        ],
        "name": "ShowFacets",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "ShowFacets"
      },
      {
        "appearance_index": 1517,
        "id": "func:django/contrib/admin/options.py:ModelAdmin.get_inline_instances:674",
        "kind": "function",
        "lines": [
          674,
          689
        ],
        "name": "ModelAdmin.get_inline_instances",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_inline_instances"
      },
      {
        "appearance_index": 1627,
        "id": "func:django/contrib/admin/options.py:BaseModelAdmin.get_inlines:397",
        "kind": "function",
        "lines": [
          397,
          399
        ],
        "name": "BaseModelAdmin.get_inlines",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_inlines"
      },
      {
        "appearance_index": 1793,
        "id": "func:django/contrib/admin/options.py:BaseModelAdmin.has_view_or_change_permission:602",
        "kind": "function",
        "lines": [
          602,
          605
        ],
        "name": "BaseModelAdmin.has_view_or_change_permission",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "has_view_or_change_permission"
      },
      {
        "appearance_index": 1867,
        "id": "func:django/contrib/admin/options.py:BaseModelAdmin.has_add_permission:545",
        "kind": "function",
        "lines": [
          545,
          552
        ],
        "name": "BaseModelAdmin.has_add_permission",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "has_add_permission"
      },
      {
        "appearance_index": 1867,
        "id": "func:django/contrib/admin/options.py:InlineModelAdmin.has_add_permission:2479",
        "kind": "function",
        "lines": [
          2479,
          2486
        ],
        "name": "InlineModelAdmin.has_add_permission",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "has_add_permission"
      },
      {
        "appearance_index": 1930,
        "id": "func:django/contrib/admin/options.py:BaseModelAdmin.has_delete_permission:569",
        "kind": "function",
        "lines": [
          569,
          582
        ],
        "name": "BaseModelAdmin.has_delete_permission",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "has_delete_permission"
      }
    ],
    "node": {
      "id": "class:django/contrib/admin/options.py:ModelAdmin:621",
      "kind": "class",
      "lines": [
        621,
        2294
      ],
      "name": "ModelAdmin",
      "path": "django/contrib/admin/options.py"
    },
    "relationship_context_policy": "consumer candidates are auto-added to W as orientation code only; read the exact node before memory_commit/repair evidence",
    "tool": "read",
    "unread_local_symbol_references": [
      {
        "appearance_index": 17,
        "id": "class:django/contrib/admin/options.py:BaseModelAdmin:124",
        "kind": "class",
        "lines": [
          124,
          618
        ],
        "name": "BaseModelAdmin",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "BaseModelAdmin"
      },
      {
        "appearance_index": 521,
        "id": "class:django/contrib/admin/options.py:ShowFacets:75",
        "kind": "class",
        "lines": [
          75,
          78
        ],
        "name": "ShowFacets",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "ShowFacets"
      },
      {
        "appearance_index": 1517,
        "id": "func:django/contrib/admin/options.py:ModelAdmin.get_inline_instances:674",
        "kind": "function",
        "lines": [
          674,
          689
        ],
        "name": "ModelAdmin.get_inline_instances",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_inline_instances"
      },
      {
        "appearance_index": 1627,
        "id": "func:django/contrib/admin/options.py:BaseModelAdmin.get_inlines:397",
        "kind": "function",
        "lines": [
          397,
          399
        ],
        "name": "BaseModelAdmin.get_inlines",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "get_inlines"
      },
      {
        "appearance_index": 1793,
        "id": "func:django/contrib/admin/options.py:BaseModelAdmin.has_view_or_change_permission:602",
        "kind": "function",
        "lines": [
          602,
          605
        ],
        "name": "BaseModelAdmin.has_view_or_change_permission",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "has_view_or_change_permission"
      },
      {
        "appearance_index": 1867,
        "id": "func:django/contrib/admin/options.py:BaseModelAdmin.has_add_permission:545",
        "kind": "function",
        "lines": [
          545,
          552
        ],
        "name": "BaseModelAdmin.has_add_permission",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "has_add_permission"
      },
      {
        "appearance_index": 1867,
        "id": "func:django/contrib/admin/options.py:InlineModelAdmin.has_add_permission:2479",
        "kind": "function",
        "lines": [
          2479,
          2486
        ],
        "name": "InlineModelAdmin.has_add_permission",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "has_add_permission"
      },
      {
        "appearance_index": 1930,
        "id": "func:django/contrib/admin/options.py:BaseModelAdmin.has_delete_permission:569",
        "kind": "function",
        "lines": [
          569,
          582
        ],
        "name": "BaseModelAdmin.has_delete_permission",
        "path": "django/contrib/admin/options.py",
        "read_status": "unread",
        "relation": "symbol_reference",
        "source": "referenced in the current implementation snippet",
        "symbol": "has_delete_permission"
      }
    ],
    "value_flow_context": [],
    "value_flow_context_policy": "value_flow_context is best-effort implementation evidence from call expressions and signatures; it shows upstream/downstream argument-to-parameter flow but is not a proof of runtime values. Related caller/callee candidates are added to W; read exact nodes before memory_commit/repair."
  },
  "step": 3,
  "visible_thinking": "The issue involves `ModelAdmin.lookup_allowed()` incorrectly raising a `DisallowedModelAdminLookup` exception when a foreign key is used as a primary key. The error occurs specifically with the lookup `restaurant__place__country`, where `place` is the primary key of `Restaurant`. The system mistakenly treats this as concrete inheritance, leading to an incorrect transformation of the lookup to `restaurant__country`, which is not in `list_filter`.\n\nFirst, I need to locate the implementation of `lookup_allowed` within the `ModelAdmin` class or its parent classes. From the initial search, I see `ModelAdmin` inherits from `BaseModelAdmin`. I should examine the `ModelAdmin` class in detail to find the `lookup_allowed` method."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "find_type": "method",
      "path_glob": "django/contrib/admin/options.py",
      "query": "lookup_allowed"
    },
    "tool": "explore_find"
  },
  "result": {
    "path_glob": "django/contrib/admin/options.py",
    "result_policy": "function/class/method/assignment results include a small implementation preview in W for orientation only; read the node before memory_commit or repair evidence. file-level results list top symbols instead of full file text",
    "results": [
      {
        "code": " 439:     def lookup_allowed(self, lookup, value):\n 440:         from django.contrib.admin.filters import SimpleListFilter\n 441: \n 442:         model = self.model\n 443:         # Check FKey lookups that are allowed, so that popups produced by\n 444:         # ForeignKeyRawIdWidget, on the basis of ForeignKey.limit_choices_to,\n 445:         # are allowed to work.\n 446:         for fk_lookup in model._meta.related_fkey_lookups:\n 447:             # As ``limit_choices_to`` can be a callable, invoke it here.\n 448:             if callable(fk_lookup):\n 449:                 fk_lookup = fk_lookup()\n 450:             if (lookup, value) in widgets.url_params_from_lookup_dict(\n 451:                 fk_lookup\n 452:             ).items():\n 453:                 return True\n 454: \n 455:         relation_parts = []\n 456:         prev_field = None\n 457:         for part in lookup.split(LOOKUP_SEP):\n 458:             try:\n 459:                 field = model._meta.get_field(part)\n 460:             except FieldDoesNotExist:\n 461:                 # Lookups on nonexistent fields are ok, since they're ignored\n 462:                 # later.\n 463:                 break\n 464:             # It is allowed to filter on values that would be found from local\n 465:             # model anyways. For example, if you filter on employee__department__id,",
        "code_preview_lines": [
          439,
          465
        ],
        "code_preview_policy": "orientation preview only; call read on this node before memory_commit or repair",
        "code_preview_truncated": true,
        "dispatch_relationship_context": [
          {
            "call": "lookup.split(LOOKUP_SEP)",
            "call_name": "lookup.split",
            "consumer_candidates": [],
            "dispatch_key_candidates": [
              {
                "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
                "key": "unknown",
                "status": "unverified_dispatch_key",
                "values": []
              }
            ],
            "dispatcher_status": "dispatcher_or_consumer_call",
            "downranked_noise": [],
            "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
            "relation": "dispatcher_context",
            "source_node": {
              "id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
              "kind": "function",
              "lines": [
                439,
                497
              ],
              "name": "BaseModelAdmin.lookup_allowed",
              "path": "django/contrib/admin/options.py"
            }
          },
          {
            "call": "fk_lookup()",
            "call_name": "fk_lookup",
            "consumer_candidates": [],
            "dispatch_key_candidates": [
              {
                "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
                "key": "unknown",
                "status": "unverified_dispatch_key",
                "values": []
              }
            ],
            "dispatcher_status": "dispatcher_or_consumer_call",
            "downranked_noise": [],
            "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
            "relation": "dispatcher_context",
            "source_node": {
              "id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
              "kind": "function",
              "lines": [
                439,
                497
              ],
              "name": "BaseModelAdmin.lookup_allowed",
              "path": "django/contrib/admin/options.py"
            }
          },
          {
            "call": "widgets.url_params_from_lookup_dict(fk_lookup).items()",
            "call_name": "widgets.url_params_from_lookup_dict(fk_lookup).items",
            "consumer_candidates": [],
            "dispatch_key_candidates": [
              {
                "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
                "key": "unknown",
                "status": "unverified_dispatch_key",
                "values": []
              }
            ],
            "dispatcher_status": "dispatcher_or_consumer_call",
            "downranked_noise": [],
            "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
            "relation": "dispatcher_context",
            "source_node": {
              "id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
              "kind": "function",
              "lines": [
                439,
                497
              ],
              "name": "BaseModelAdmin.lookup_allowed",
              "path": "django/contrib/admin/options.py"
            }
          },
          {
            "call": "widgets.url_params_from_lookup_dict(fk_lookup)",
            "call_name": "widgets.url_params_from_lookup_dict",
            "consumer_candidates": [],
            "dispatch_key_candidates": [
              {
                "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
                "key": "unknown",
                "status": "unverified_dispatch_key",
                "values": []
              }
            ],
            "dispatcher_status": "dispatcher_or_consumer_call",
            "downranked_noise": [],
            "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
            "relation": "dispatcher_context",
            "source_node": {
              "id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
              "kind": "function",
              "lines": [
                439,
                497
              ],
              "name": "BaseModelAdmin.lookup_allowed",
              "path": "django/contrib/admin/options.py"
            }
          }
        ],
        "dispatch_tables": [],
        "id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
        "kind": "function",
        "lines": [
          439,
          497
        ],
        "local_symbol_references": [],
        "name": "BaseModelAdmin.lookup_allowed",
        "path": "django/contrib/admin/options.py",
        "score": 4.0,
        "source": "graph",
        "unread_local_symbol_references": []
      }
    ],
    "tool": "explore_find",
    "warning": null
  },
  "step": 4,
  "visible_thinking": "The issue is about `ModelAdmin.lookup_allowed()` incorrectly raising `DisallowedModelAdminLookup` when a foreign key is used as a primary key. The problem seems to stem from the admin thinking that having a foreign key field as a primary key is the same as concrete inheritance, leading to incorrect lookup handling.\n\nFrom the failing test case:\n- `Restaurant` has a `OneToOneField` to `Place` with `primary_key=True`\n- `Waiter` has a `ForeignKey` to `Restaurant`\n- The `list_filter` includes `'restaurant__place__country'`\n- The `lookup_allowed` method is expected to return `True` for this lookup, but it's returning `False`\n\nLooking at the `ModelAdmin` class that was read, I need to find the `lookup_allowed` method. Since the read was truncated, I should look for this method specifically.\n\nLet me search for the `lookup_allowed` method in the `ModelAdmin` class."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "node_id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
      "view": "body"
    },
    "tool": "read"
  },
  "result": {
    "code": " 439:     def lookup_allowed(self, lookup, value):\n 440:         from django.contrib.admin.filters import SimpleListFilter\n 441: \n 442:         model = self.model\n 443:         # Check FKey lookups that are allowed, so that popups produced by\n 444:         # ForeignKeyRawIdWidget, on the basis of ForeignKey.limit_choices_to,\n 445:         # are allowed to work.\n 446:         for fk_lookup in model._meta.related_fkey_lookups:\n 447:             # As ``limit_choices_to`` can be a callable, invoke it here.\n 448:             if callable(fk_lookup):\n 449:                 fk_lookup = fk_lookup()\n 450:             if (lookup, value) in widgets.url_params_from_lookup_dict(\n 451:                 fk_lookup\n 452:             ).items():\n 453:                 return True\n 454: \n 455:         relation_parts = []\n 456:         prev_field = None\n 457:         for part in lookup.split(LOOKUP_SEP):\n 458:             try:\n 459:                 field = model._meta.get_field(part)\n 460:             except FieldDoesNotExist:\n 461:                 # Lookups on nonexistent fields are ok, since they're ignored\n 462:                 # later.\n 463:                 break\n 464:             # It is allowed to filter on values that would be found from local\n 465:             # model anyways. For example, if you filter on employee__department__id,\n 466:             # then the id value would be found already from employee__department_id.\n 467:             if not prev_field or (\n 468:                 prev_field.is_relation\n 469:                 and field not in prev_field.path_infos[-1].target_fields\n 470:             ):\n 471:                 relation_parts.append(part)\n 472:             if not getattr(field, \"path_infos\", None):\n 473:                 # This is not a relational field, so further parts\n 474:                 # must be transforms.\n 475:                 break\n 476:             prev_field = field\n 477:             model = field.path_infos[-1].to_opts.model\n 478: \n 479:         if len(relation_parts) <= 1:\n 480:             # Either a local field filter, or no fields at all.\n 481:             return True\n 482:         valid_lookups = {self.date_hierarchy}\n 483:         for filter_item in self.list_filter:\n 484:             if isinstance(filter_item, type) and issubclass(\n 485:                 filter_item, SimpleListFilter\n 486:             ):\n 487:                 valid_lookups.add(filter_item.parameter_name)\n 488:             elif isinstance(filter_item, (list, tuple)):\n 489:                 valid_lookups.add(filter_item[0])\n 490:             else:\n 491:                 valid_lookups.add(filter_item)\n 492: \n 493:         # Is it a valid relational lookup?\n 494:         return not {\n 495:             LOOKUP_SEP.join(relation_parts),\n 496:             LOOKUP_SEP.join(relation_parts + [part]),\n 497:         }.isdisjoint(valid_lookups)",
    "dispatch_relationship_context": [
      {
        "call": "lookup.split(LOOKUP_SEP)",
        "call_name": "lookup.split",
        "consumer_candidates": [],
        "dispatch_key_candidates": [
          {
            "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
            "key": "unknown",
            "status": "unverified_dispatch_key",
            "values": []
          }
        ],
        "dispatcher_status": "dispatcher_or_consumer_call",
        "downranked_noise": [],
        "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
        "relation": "dispatcher_context",
        "source_node": {
          "id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
          "kind": "function",
          "lines": [
            439,
            497
          ],
          "name": "BaseModelAdmin.lookup_allowed",
          "path": "django/contrib/admin/options.py"
        }
      },
      {
        "call": "{LOOKUP_SEP.join(relation_parts), LOOKUP_SEP.join(relation_parts + [part])}.isdisjoint(valid_lookups)",
        "call_name": "{LOOKUP_SEP.join(relation_parts), LOOKUP_SEP.join(relation_parts + [part])}.isdisjoint",
        "consumer_candidates": [],
        "dispatch_key_candidates": [
          {
            "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
            "key": "unknown",
            "status": "unverified_dispatch_key",
            "values": []
          }
        ],
        "dispatcher_status": "dispatcher_or_consumer_call",
        "downranked_noise": [],
        "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
        "relation": "dispatcher_context",
        "source_node": {
          "id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
          "kind": "function",
          "lines": [
            439,
            497
          ],
          "name": "BaseModelAdmin.lookup_allowed",
          "path": "django/contrib/admin/options.py"
        }
      },
      {
        "call": "fk_lookup()",
        "call_name": "fk_lookup",
        "consumer_candidates": [],
        "dispatch_key_candidates": [
          {
            "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
            "key": "unknown",
            "status": "unverified_dispatch_key",
            "values": []
          }
        ],
        "dispatcher_status": "dispatcher_or_consumer_call",
        "downranked_noise": [],
        "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
        "relation": "dispatcher_context",
        "source_node": {
          "id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
          "kind": "function",
          "lines": [
            439,
            497
          ],
          "name": "BaseModelAdmin.lookup_allowed",
          "path": "django/contrib/admin/options.py"
        }
      },
      {
        "call": "widgets.url_params_from_lookup_dict(fk_lookup).items()",
        "call_name": "widgets.url_params_from_lookup_dict(fk_lookup).items",
        "consumer_candidates": [],
        "dispatch_key_candidates": [
          {
            "evidence": "current call looks like a dispatcher, but no issue literal was bound to a key",
            "key": "unknown",
            "status": "unverified_dispatch_key",
            "values": []
          }
        ],
        "dispatcher_status": "dispatcher_or_consumer_call",
        "downranked_noise": [],
        "planner_guidance": "Treat these as relation facts and next-read candidates, not final targets. If a wrapper/dispatcher is read, close the chain by reading the actual consumer before repair.",
        "relation": "dispatcher_context",
        "source_node": {
          "id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
          "kind": "function",
          "lines": [
            439,
            497
          ],
          "name": "BaseModelAdmin.lookup_allowed",
          "path": "django/contrib/admin/options.py"
        }
      }
    ],
    "dispatch_tables": [],
    "local_symbol_references": [],
    "node": {
      "id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
      "kind": "function",
      "lines": [
        439,
        497
      ],
      "name": "BaseModelAdmin.lookup_allowed",
      "path": "django/contrib/admin/options.py"
    },
    "relationship_context_policy": "consumer candidates are auto-added to W as orientation code only; read the exact node before memory_commit/repair evidence",
    "tool": "read",
    "unread_local_symbol_references": [],
    "value_flow_context": [],
    "value_flow_context_policy": "value_flow_context is best-effort implementation evidence from call expressions and signatures; it shows upstream/downstream argument-to-parameter flow but is not a proof of runtime values. Related caller/callee candidates are added to W; read exact nodes before memory_commit/repair."
  },
  "step": 5,
  "visible_thinking": "The issue involves `ModelAdmin.lookup_allowed()` incorrectly raising `DisallowedModelAdminLookup` when a foreign key is used as a primary key. The problem seems to stem from the admin confusing this scenario with concrete inheritance, leading to incorrect lookup handling.\nFrom the exploration so far, I've identified the `ModelAdmin` class and the `lookup_allowed` method in `BaseModelAdmin`. The `lookup_allowed` method is where the logic for validating lookups resides. Given the issue description, the problem likely lies in how the method processes lookups involving foreign keys that are primary keys.\nSince the `ModelAdmin` class was read, and the `lookup_allowed` method was found, the next step is to read the full `lookup_allowed` method to understand its implementation and identify where the issue might be occurring. The method is currently only available as a preview.\nOnce the method is fully read, I can analyze the logic and determine the necessary fix. Since the repair mechanism is currently blocked due to lack of hydrated memory, I'll need to commit the relevant nodes to the repair memory after reading them.\nGiven the current state, the most logical next step is to read the `lookup_allowed` method in full."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "note": "This is the function where the lookup validation issue occurs, as described in the bug report.",
      "select_ids": [
        "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439"
      ]
    },
    "tool": "memory_commit"
  },
  "result": {
    "already_present_ids": [],
    "committed": [
      "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439"
    ],
    "dropped_by_keep_ids": [],
    "explicitly_selected_ids": [
      "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439"
    ],
    "memory": [
      {
        "has_code": true,
        "id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
        "kind": "function",
        "lines": [
          439,
          497
        ],
        "name": "BaseModelAdmin.lookup_allowed",
        "path": "django/contrib/admin/options.py"
      }
    ],
    "memory_changed": true,
    "newly_added_ids": [
      "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439"
    ],
    "tool": "memory_commit"
  },
  "step": 6,
  "visible_thinking": "The issue involves `ModelAdmin.lookup_allowed()` incorrectly raising a `DisallowedModelAdminLookup` when a foreign key is used as a primary key. The problem seems to stem from the method confusing this scenario with concrete inheritance, leading to incorrect lookup validation.\nFrom the code read, I see that `BaseModelAdmin.lookup_allowed` is the method where the issue occurs. The logic in this method handles relation parts and validates lookups against `list_filter`. The issue description suggests that the method incorrectly shortcuts lookups when a foreign key is a primary key, treating it as concrete inheritance.\nI need to commit the read code for `BaseModelAdmin.lookup_allowed` to the repair memory, as it's directly involved in the issue. The `ModelAdmin` class is also relevant as it's the parent class, but since the issue is specifically in the `lookup_allowed` method, I'll focus on that.\nLet me check if there are any other relevant pieces of code that need to be understood or committed. The issue mentions a test case, but test source isn't available for repair evidence. However, the runtime failure shows that the assertion `self.assertIs(ma.lookup_allowed('restaurant__place__country', 'test_value'), True)` fails, meaning `lookup_allowed` returns `False` when it should return `True`.\nLooking at the code, the key part is how `relation_parts` are constructed and validated. The issue likely lies in how the method determines whether a field sequence is valid, particularly when dealing with foreign keys that are primary keys.\nSince I've read the necessary code and identified the problematic function, I should commit it to the repair memory to enable repair actions."
}
```

## planner_step

```json
{
  "action": {
    "params": {
      "confidence": 0.8,
      "evidence_chain": [
        {
          "evidence": "This is the function that incorrectly rejects the lookup for foreign key chains involving OneToOneField with primary_key=True",
          "node_id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
          "role": "target"
        }
      ],
      "failure_seen": "False is not True when calling ma.lookup_allowed('restaurant__place__country', 'test_value')",
      "intent_analysis": "The issue is in the lookup_allowed method where it incorrectly determines relation_parts for lookups involving OneToOneField with primary_key=True. The method treats these as concrete inheritance cases and skips intermediate relations, causing valid lookups like 'restaurant__place__country' to be shortened to 'restaurant__country' which doesn't match the declared list_filter.",
      "target_nodes": [
        "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439"
      ]
    },
    "tool": "repair_review"
  },
  "result": {
    "cgm_payload": {
      "cgm_insight_count": 0,
      "dispatch_table_count": 0,
      "edge_count": 7,
      "has_adjacency_list": true,
      "has_prompt": true,
      "has_subgraph": true,
      "issue_body_chars": 2637,
      "issue_id": "django__django-16661",
      "issue_title_present": true,
      "node_count": 8,
      "node_paths": [
        "django/contrib/admin/filters.py",
        "django/contrib/admin/options.py",
        "django/core/exceptions.py",
        "django/db/models/constants.py"
      ],
      "pending_patch_present": false,
      "plan_target_count": 1,
      "plan_targets": [
        "django/contrib/admin/options.py:439-497"
      ],
      "plan_text_chars": 1463,
      "planner_confidence": 0.8,
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
      "output_format": "intent_review_json",
      "raw_preview": "{\n  \"verdict\": \"ready\",\n  \"confidence\": 0.90,\n  \"mechanism_assessment\": \"The condition at lines 467-470 incorrectly excludes intermediate model fields when processing OneToOneField primary keys, causing lookup chains like 'restaurant__place__country' to be shortened to 'restaurant__country' which doesn't match list_filter declarations.\",\n  \"target_assessment\": \"The entire lookup_allowed function is the correct target since the bug originates in its relation_parts building logic when handling primary-key OneToOneFields, as demonstrated by the test case failure within this function's execution.\",\n  \"evidence_gaps\": [],\n  \"suggested_next_action\": \"Modify the condition at lines 467-470 in django/contrib/admin/options.py to prevent skipping OneToOneField primary key relations during lookup chai",
      "reasoning_chars": 23806,
      "reasoning_preview": "We are reviewing a proposed repair intent for a bug in Django's ModelAdmin.lookup_allowed() method.\n\nIssue Summary:\n- The bug: ModelAdmin.lookup_allowed() incorrectly raises DisallowedModelAdminLookup for a lookup chain that includes a foreign key as a primary key (specifically, a OneToOneField with primary_key=True).\n- The test case: \n    - Models: Country -> Place (FK to Country) -> Restaurant (OneToOne to Place, primary_key=True) -> Waiter (FK to Restaurant)\n    - WaiterAdmin has list_filter = ['restaurant__place__country']\n    - The test checks: ma.lookup_allowed('restaurant__place__country', 'test_value') should be True, but it returns False (hence the failure).\n- The reporter's hypothesis: The admin code mistakenly treats the OneToOneField (with primary_key=True) as if it were concrete inheritance, so when processing the lookup 'restaurant__place__country', it shortcuts to 'restaurant__country' (which is not in list_filter).\n\nRelevant Code (django/contrib/admin/options.py:439-497):\nThe function `lookup_allowed` is provided. Key parts:\n- It splits the lookup string by `LOOKUP_SEP` (usually '__') and traverses the model relations.\n- For each part, it gets the field from the cur",
      "thinking_enabled": true
    },
    "note_to_planner": "Use this CGM critique to revise intent_analysis, target_nodes, evidence_chain, memory_delete stale nodes, or proceed to repair if verdict/confidence support the current mechanism.",
    "review": {
      "adoption_advice": "Adopt the planner intent to repair the lookup_allowed function, as the issue text and code snippet conclusively show the faulty relation_parts logic for primary-key OneToOneFields.",
      "adoption_caveat": "",
      "confidence": 0.9,
      "evidence_gaps": [],
      "mechanism_assessment": "The condition at lines 467-470 incorrectly excludes intermediate model fields when processing OneToOneField primary keys, causing lookup chains like 'restaurant__place__country' to be shortened to 'restaurant__country' which doesn't match list_filter declarations.",
      "removed_benchmark_test_source_requests": [],
      "suggested_next_action": "Modify the condition at lines 467-470 in django/contrib/admin/options.py to prevent skipping OneToOneField primary key relations during lookup chain validation.",
      "summary": "dashscope-cgm-bridge-review",
      "target_assessment": "The entire lookup_allowed function is the correct target since the bug originates in its relation_parts building logic when handling primary-key OneToOneFields, as demonstrated by the test case failure within this function's execution.",
      "verdict": "ready"
    },
    "status": "reviewed",
    "tool": "repair_review"
  },
  "step": 7,
  "visible_thinking": "Let me analyze the issue:\n\n1. The problem is in `ModelAdmin.lookup_allowed()` method which incorrectly raises `DisallowedModelAdminLookup` when dealing with foreign keys as primary keys.\n2. The issue description suggests that the admin thinks having a foreign key field as a primary key is the same as concrete inheritance.\n3. When checking lookups for `restaurant__place__country`, it thinks 'place' is the concrete parent of 'restaurant' and shortcuts it to `restaurant__country` which isn't in `list_filter`.\n\nLooking at the code for `lookup_allowed` function (lines 439-497), I can see the logic that processes the lookup parts and builds `relation_parts`. The issue seems to be in how it handles the relationship traversal when there's a OneToOneField with primary_key=True.\n\nLet me examine the key part of the logic:\n- Lines 455-477 process each part of the lookup\n- Lines 467-470 determine whether to add a part to `relation_parts`\n- Lines 494-496 check if the constructed lookup is valid against `valid_lookups`\n\nThe problem appears to be in the condition on lines 467-470:\n```python\nif not prev_field or (\n    prev_field.is_relation\n    and field not in prev_field.path_infos[-1].target_fields\n):\n```\n\nThis condition is meant to handle cases where we're looking at a field that's already covered by a previous relation, but it's not correctly handling the case where we have a OneToOneField as primary key.\n\nLet me check if I need to read more code to understand the path_infos structure better, or if I can proceed with the repair."
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
          "evidence": "This is the function that incorrectly rejects the lookup for foreign key chains involving OneToOneField with primary_key=True",
          "node_id": "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439",
          "role": "target"
        }
      ],
      "failure_seen": "False is not True when calling ma.lookup_allowed('restaurant__place__country', 'test_value')",
      "intent_analysis": "The issue is in the lookup_allowed method where it incorrectly determines relation_parts for lookups involving OneToOneField with primary_key=True. The method treats these as concrete inheritance cases and skips intermediate relations, causing valid lookups like 'restaurant__place__country' to be shortened to 'restaurant__country' which doesn't match the declared list_filter. The fix involves modifying the condition at lines 467-470 to correctly handle OneToOneField primary key relations.",
      "target_nodes": [
        "func:django/contrib/admin/options.py:BaseModelAdmin.lookup_allowed:439"
      ]
    },
    "tool": "repair"
  },
  "result": {
    "cgm_payload": {
      "cgm_insight_count": 0,
      "dispatch_table_count": 0,
      "edge_count": 7,
      "has_adjacency_list": true,
      "has_prompt": true,
      "has_subgraph": true,
      "issue_body_chars": 2637,
      "issue_id": "django__django-16661",
      "issue_title_present": true,
      "node_count": 8,
      "node_paths": [
        "django/contrib/admin/filters.py",
        "django/contrib/admin/options.py",
        "django/core/exceptions.py",
        "django/db/models/constants.py"
      ],
      "pending_patch_present": false,
      "plan_target_count": 1,
      "plan_targets": [
        "django/contrib/admin/options.py:439-497"
      ],
      "plan_text_chars": 1571,
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
      "raw_preview": "{\n  \"summary\": \"Fix lookup_allowed to include relation fields in relation_parts when they are part of the target of the previous relation, preventing incorrect skipping of intermediate models in lookup chains involving OneToOneField primary keys.\",\n  \"edits\": [\n    {\n      \"path\": \"django/contrib/admin/options.py\",\n      \"start\": 469,\n      \"end\": 469,\n      \"new_text\": \"                and (field.is_relation or field not in prev_field.path_infos[-1].target_fields)\"\n    }\n  ]\n}",
      "reasoning_chars": 26134,
      "reasoning_preview": "We are given an issue: ModelAdmin.lookup_allowed() incorrectly raises DisallowedModelAdminLookup for a lookup that involves a foreign key as primary key (specifically, a OneToOneField with primary_key=True).\n\nThe problem is that in the lookup_allowed method, when processing a lookup string (like 'restaurant__place__country'), the code tries to break it down into relation parts. The issue arises because when a model has a OneToOneField as its primary key (like Restaurant having a OneToOneField to Place as primary key), the admin code mistakenly treats the OneToOneField as if it were a concrete inheritance parent. This causes the lookup to skip the intermediate model (Place) and jump to the next level (so 'restaurant__place__country' becomes 'restaurant__country').\n\nLooking at the code in django/contrib/admin/options.py:439-497:\n\nThe relevant part is the loop that breaks the lookup string into parts (lines 455-478). In particular, the condition at lines 467-470:\n\n  467:             if not prev_field or (\n  468:                 prev_field.is_relation\n  469:                 and field not in prev_field.path_infos[-1].target_fields\n  470:             ):\n  471:                 relation_pa",
      "thinking_enabled": true
    },
    "done": true,
    "patch_preview": {
      "edit_count": 1,
      "edits": [
        {
          "end": 469,
          "new_text": "and (field.is_relation or field not in prev_field.path_infos[-1].target_fields)",
          "path": "django/contrib/admin/options.py",
          "start": 469
        }
      ],
      "summary": "Fix lookup_allowed to include relation fields in relation_parts when they are part of the target of the previous relation, preventing incorrect skipping of intermediate models in lookup chains involving OneToOneField primary keys."
    },
    "rolled_back": false,
    "status": "passed",
    "summary": "Fix lookup_allowed to include relation fields in relation_parts when they are part of the target of the previous relation, preventing incorrect skipping of intermediate models in lookup chains involving OneToOneField primary keys.",
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
    "tool": "repair",
    "touched_paths": [
      "django/contrib/admin/options.py"
    ]
  },
  "step": 8,
  "visible_thinking": "The repair review has concluded that the issue is in the `lookup_allowed` method, specifically in how it handles OneToOneField with primary_key=True. The review suggests modifying the condition at lines 467-470 to prevent skipping these relations. I will now proceed to implement this repair."
}
```
