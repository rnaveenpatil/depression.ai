# depression.ai — Project File Listing

Generated: 2026-10-21 21:47:27
Repo: 4094369  Branch: main

## Summary

| Area | Files | Size |
|---|---:|---:|
| `src/agent` | 107 | 89M |
| `tests` | 25 | 820K |
| `docs` | 12 | 2.1M |
| `assets` | 37 | 5.8M |
| `.opencode` | 3 | 63M |
| `.github` | 3 | 24K |
| `.githooks` | 1 | 12K |
| `.agent` | 3 | 20K |
| (root files) | 13 | — |
| **total** | **204** | |

Excluded: .git/, deprssion/ (venv), dist/, .opencode/node_modules/,
src/agent/depression/ (nested venv), .mypy_cache/, .pytest_cache/,
.ruff_cache/, __pycache__/

## Files with sizes (bytes)

```
         0  .agent/cache/.gitkeep
       185  .agent/config.json
         0  .agent/sessions/.gitkeep
     79974  assets/AWS Authentication And Account Access.png
    119063  assets/Command Reference Interface.png
     74053  assets/LLM Connection Configuration.png
     73281  assets/llm_connect.png
    276779  assets/LLM Usage and Cost Tracking.png
       820  assets/main project - Repository debugging analysis_files/ansi-1f6vhsjh.css
       586  assets/main project - Repository debugging analysis_files/app-block-trigger-actions-hxl87txx.css
      3060  assets/main project - Repository debugging analysis_files/client-defined-widget-gdqdmhro.css
      1916  assets/main project - Repository debugging analysis_files/code-block-dop2czwo.css
       962  assets/main project - Repository debugging analysis_files/code-block-viewer-bb2wmqiy.css
    184852  assets/main project - Repository debugging analysis_files/conversation-small-ou5gr6jt(1).css
    184852  assets/main project - Repository debugging analysis_files/conversation-small-ou5gr6jt.css
       248  assets/main project - Repository debugging analysis_files/cot-message-chunk-b8hse3v2.css
      1320  assets/main project - Repository debugging analysis_files/cot-message-kqp4eshj.css
      2466  assets/main project - Repository debugging analysis_files/cot-v5-cwqe3unp.css
      1345  assets/main project - Repository debugging analysis_files/FormattedText-dn9bmv7i.css
     52764  assets/main project - Repository debugging analysis_files/global-modals-m6d51e47.css
     11205  assets/main project - Repository debugging analysis_files/icon_dark.png
      5340  assets/main project - Repository debugging analysis_files/image-viewer-f73z3svd.css
      3584  assets/main project - Repository debugging analysis_files/multiplayer-pp910s5j.css
     20446  assets/main project - Repository debugging analysis_files/react-cj2gxmrc.css
   1858957  assets/main project - Repository debugging analysis_files/root-in2rmu2p.css
      8585  assets/main project - Repository debugging analysis_files/root-nhnbl08z(1).css
      8585  assets/main project - Repository debugging analysis_files/root-nhnbl08z.css
      1259  assets/main project - Repository debugging analysis_files/rp.png
       537  assets/main project - Repository debugging analysis_files/table-components-b39fqymt.css
      1427  assets/main project - Repository debugging analysis_files/writing-block-gwgymy54.css
     25481  assets/main project - Repository debugging analysis_files/writing-block-provider-lbnz1gc0.css
   1672803  assets/meghasree.png
     83671  assets/mit_logo.png
    831244  assets/patil.png
    115380  assets/rahul.jpeg
     27582  assets/sumantha.jpeg
     37006  assets/Tool Permission and  Execution Confirmation.png
     53560  assets/tui.png
     55790  assets/User Interface (1).png
     94051  assets/User Profile.png
     73281  docs/assets/llm_connect.png
    822968  docs/assets/meghasree.png
     83671  docs/assets/mit_logo.png
    831244  docs/assets/patil.png
    115380  docs/assets/rahul.jpeg
     27582  docs/assets/sumantha.jpeg
     53560  docs/assets/tui.png
     15069  docs/context-bugs-report.txt
     13139  docs/credential-exposure-report.md
     10040  docs/firebase-onboarding.md
     50819  docs/index.html
      2564  docs/opencode-runtime-parity-report.md
       261  .env
       283  .env.example
      6080  .githooks/pre-commit
       892  .github/workflows/publish.yml
      8130  .github/workflows/secret-scan.yml
       538  .github/workflows/tui-test.yml
       763  .gitignore
      5382  .gitleaks.toml
      1198  MANIFEST.in
        63  .opencode/.gitignore
        65  .opencode/package.json
     14634  .opencode/package-lock.json
      2017  pyproject.toml
     50590  README.md
     28156  src/agent/agent/agent.py
     43887  src/agent/agent/dual_agent.py
      1124  src/agent/agent/__init__.py
    117298  src/agent/agent/loop.py
     35888  src/agent/agent/planner.py
     18286  src/agent/agent/subagent.py
     74314  src/agent/cli/commands.py
       603  src/agent/cli/__init__.py
     35789  src/agent/cli/input.py
     24819  src/agent/cli/ui.py
     37283  src/agent/config/config.py
      3936  src/agent/config/__init__.py
     27544  src/agent/config/loader.py
     17303  src/agent/context/compaction.py
     12513  src/agent/context/files.py
       797  src/agent/context/__init__.py
     32790  src/agent/context/manager.py
     10844  src/agent/context/project.py
     16850  src/agent/context/runtime.py
      4588  src/agent/context/summary_store.py
       983  src/agent/__init__.py
     43161  src/agent/llm/anthropic.py
     46445  src/agent/llm/gemini.py
      3314  src/agent/llm/__init__.py
     36764  src/agent/llm/normalizer.py
     43772  src/agent/llm/ollama.py
     38193  src/agent/llm/openai_compatible.py
     20622  src/agent/llm/provider.py
      3390  src/agent/llm/rate_limiter.py
     49752  src/agent/llm/runtime.py
     30019  src/agent/main.py
      7526  src/agent/mcp/aws_config.py
     58358  src/agent/mcp/client.py
      1745  src/agent/mcp/__init__.py
      9095  src/agent/permissions/aws.py
      7857  src/agent/permissions/filesystem.py
       662  src/agent/permissions/__init__.py
     29047  src/agent/permissions/manager.py
      9682  src/agent/permissions/rules.py
     10882  src/agent/permissions/terminal.py
      1045  src/agent/plugins/__init__.py
     24062  src/agent/plugins/loader.py
     14126  src/agent/plugins/registry.py
      9952  src/agent/project/awsproject.py
     15762  src/agent/project/git.py
       799  src/agent/project/__init__.py
     15941  src/agent/project/scanner.py
     19810  src/agent/project/workspace.py
     10632  src/agent/session/history.py
       684  src/agent/session/__init__.py
     21693  src/agent/session/session.py
      3309  src/agent/session/state.py
     16279  src/agent/storage/cache.py
     22537  src/agent/storage/database.py
       479  src/agent/storage/__init__.py
     15176  src/agent/tools/aws_helper.py
      4789  src/agent/tools/browser.py
     32869  src/agent/tools/compat.py
      4033  src/agent/tools/diagnostics.py
     23916  src/agent/tools/filesystem.py
      5255  src/agent/tools/git.py
      1638  src/agent/tools/__init__.py
      2474  src/agent/tools/mcp.py
      9711  src/agent/tools/patch.py
      2531  src/agent/tools/path_guard.py
     16554  src/agent/tools/proccess.py
     28567  src/agent/tools/realtime_browser.py
     19816  src/agent/tools/registry.py
      6845  src/agent/tools/search.py
      4493  src/agent/tools/task.py
     11883  src/agent/tools/terminal.py
      5672  src/agent/tools/todo.py
     11049  src/agent/tools/web.py
     91239  src/agent/tui/app.py
      5688  src/agent/tui/events.py
       124  src/agent/tui/__init__.py
       113  src/agent/tui/__main__.py
     15365  src/agent/tui/onboarding/empty_banner.py
     29233  src/agent/tui/onboarding/firebase_auth.py
     15607  src/agent/tui/onboarding/firebase_config.py
       236  src/agent/tui/onboarding/firebase_default.json
     13568  src/agent/tui/onboarding/google_oauth.py
      9150  src/agent/tui/onboarding/identity.py
      1842  src/agent/tui/onboarding/__init__.py
      9656  src/agent/tui/onboarding/profile.py
     20076  src/agent/tui/onboarding/welcome.py
      6897  src/agent/tui/plan_parser.py
      3412  src/agent/tui/run.py
      1160  src/agent/tui/theme.py
      7097  src/agent/tui/tool_call.py
     29316  src/agent/tui/widgets/aws_panel.py
      5888  src/agent/tui/widgets/diff_view.py
     11647  src/agent/tui/widgets/empty_banner.py
     15082  src/agent/tui/widgets/output_view.py
      6244  src/agent/tui/widgets/plan_view.py
      4519  src/agent/tui/widgets/profile_panel.py
      2812  src/agent/tui/widgets/sidebar.py
      1857  src/agent/tui/widgets/thinking.py
      8163  src/agent/tui/widgets/todo_panel.py
     26497  src/agent/tui/widgets/tool_call.py
     16396  src/agent/utils/env_manager.py
     17871  src/agent/utils/errors.py
      2710  src/agent/utils/__init__.py
      7498  src/agent/utils/logging.py
      7170  src/agent/utils/platform.py
      7581  src/agent/utils/redact.py
      2601  src/agent/_version.py
         1  src/depression.ai.egg-info/dependency_links.txt
       115  src/depression.ai.egg-info/entry_points.txt
     44771  src/depression.ai.egg-info/PKG-INFO
       279  src/depression.ai.egg-info/requires.txt
      3425  src/depression.ai.egg-info/SOURCES.txt
         6  src/depression.ai.egg-info/top_level.txt
        48  tests/__init__.py
         0  tests/test_agent.py
      8080  tests/test_agents_roles.py
     10845  tests/test_aws_panel_save.py
     10398  tests/test_base_url_adaptation.py
         0  tests/test_cli.py
         0  tests/test_config.py
      1451  tests/test_context_runtime.py
      2552  tests/test_filesystem_write.py
      9268  tests/test_llm_adapters_http.py
     11131  tests/test_llm_tool_calls.py
      4922  tests/test_loop_speedups.py
     49698  tests/test_onboarding.py
      6959  tests/test_permission_actions.py
      4540  tests/test_redaction.py
      3011  tests/test_registry_normalization.py
      1610  tests/test_runtime_connection.py
         0  tests/test_session.py
      1946  tests/test_shutdown.py
      9694  tests/test_tool_call_history.py
         0  tests/test_tools.py
      2498  tests/test_tui_llm_chain.py
      3500  tests/test_tui_pty.py
      7597  tests/test_tui.py
      3948  tests/test_version.py
```
