# Credential Exposure & Security Report

**Repository:** `github.com/rnaveenpatil/depression.ai`
**Visibility:** **PUBLIC** (verified: `https://github.com/rnaveenpatil/depression.ai` returns HTTP 200 unauthenticated)
**Audit date:** 2026-10-05
**Audited ref:** local `main` = `4afefa4`; `origin/main` = `f3ffed8` (local is behind)

---

## 1. Credential Exposure Report

Three real credentials were committed and are retrievable from git history by
anyone, because the repository is public. Two are still live on `origin/main`.

### Credential 1 — Firebase / Google API key (**STILL LIVE**)

| Field | Value |
|---|---|
| Key | `AIzaSyAZxE…qE7QXmk  (prefix/suffix only)` |
| Type | Google Cloud API key (Firebase web client) |
| Project | `cloudsever-526fb` |
| Project number | `605436333256` |
| Storage bucket | `cloudsever-526fb.firebasestorage.app` |
| OAuth client (web) | `605436333256-…ae4dh1.apps.googleusercontent.com` |
| OAuth client (Android) | `605436333256-…r2f7b43.apps.googleusercontent.com` |
| Android package | `com.cloud_saver` |

**Where it was exposed**

| Location | Introduced | Present on `origin/main` |
|---|---|---|
| `tests/test_onboarding.py:1107` (`current_key`) | commit `262d5ce` "almost fixed and added gmail login" | **Yes — live** |
| `tests/test_onboarding.py:1117` (`assert cfg.api_key == ...`) | commit `262d5ce` | **Yes — live** |
| `src/agent/tui/onboarding/firebase_config.py:23` (project number) | — | Yes |
| `docs/firebase-onboarding.md:75-78` (project id, number, bucket) | — | Yes |

**Blast radius.** A Firebase *web* API key is not a secret in the same sense as
a password — it is designed to ship in client-side code — but it is the key that
authorises calls to the project's Firebase services. Exposure matters because:

1. It is paired with the **OAuth client id**, which together are exactly the
   two values the sign-in flow consumes. Anyone can impersonate this
   application's Firebase identity and, if the key is unrestricted, enumerate
   or read data reachable by that key.
2. Firebase keys are commonly left **unrestricted** in the console. If that is
   the case here, the key can be used to read/write Firestore, Storage, or Auth
   data belonging to project `cloudsever-526fb`.
3. It identifies a real, live production project to attackers.

**Required action (do this first):**
- Google Cloud Console → **APIs & Services → Credentials** for project
  `cloudsever-526fb` → delete/replace this key.
- **Restrict any replacement key** by API and by referring service. Never ship
  an unrestricted Firebase key.
- Add HTTP referrer restrictions for web keys.
- Confirm whether project data was accessed from unexpected IPs (Cloud
  Logging, Auth usage reports).

### Credential 2 — Groq API key (in history)

| Field | Value |
|---|---|
| Key | `gsk_YKC0…wTHBAOk  (prefix/suffix only)` |
| Type | Groq LLM API key |
| Constant | `HARDCODED_API_KEY` in `src/agent/llm/groq.py` |
| Also referenced in | `src/agent/llm/nvidia.py` error messages (pattern, no value) |

**Where it was exposed.** Committed as a hardcoded fallback in
`src/agent/llm/groq.py`, which was itself used as a *fallback* when no
`GROQ_API_KEY` environment variable was set. Present in history across commits
including `b5c0946`, `ba1cf57`, `5ebd519`, `e026b53`.

**Status now.** Removed from the working tree by commit `8a16870` ("fix all
credentials essiue") and the entire `groq.py` file has since been deleted as
dead code (see §3). Still fully retrievable from history.

**Blast radius.** Allows billable LLM inference on the owner's Groq account,
and free-tier abuse / rate-limit exhaustion against it.

**Required action.** Revoke the key in the Groq console, then rotate.

### Credential 3 — OpenAI / ChatGPT account access token (in history)

| Field | Value |
|---|---|
| Type | Full JWT access token (ChatGPT account credentials) |
| Header | `{"alg":"RS256","kid":"n0z6Pr1-…8kRVs","typ":"JWT"}` |
| Audience | `https://api.openai.com/v1` |
| Client id | `app_X8zY…L5gH` |
| Account id | `dcf95921-…031a3d5` |
| User ids | `user-QWMHp…h9JBQL…` |
| Plan | `chatgpt_plan_type: free` |

**Where it was exposed.** Inside a browser debugging artifact committed to the
repository:

```
assets/main project - Repository debugging analysis.html
assets/main project - Repository debugging analysis.pdf
```

These were captured devtools/network exports — 898 lines each — containing a
live session token in a request/response dump. Deleted in commit `8a16870`.

**Blast radius.** This is the most serious of the three. It is a bearer token
for an authenticated OpenAI account session, not an API key. It grants access
to the account behind it (conversation history, and whatever the ChatGPT
session permits) until revoked.

**Required action.** Revoke all active ChatGPT sessions / re-authenticate on
every account whose token appeared, and review account activity.

### Not exposed (verified clean)

| Check | Result |
|---|---|
| AWS credentials in git history | **None.** Scanned all 1,650 blobs for `AKIA…` — zero hits |
| AWS credentials in `.env` | Present on disk, correctly gitignored, **never committed** (`git log -S` returns nothing) |
| GitHub PAT / classic or fine-grained | None |
| Slack, npm, Stripe, GitLab, Discord tokens | None |
| PEM / OpenSSH private keys | None |
| `.env` tracked by git | No — `.gitignore` is correct |
| `google-services.json` / `firebase.json` / service accounts | None committed |

---

## 2. Framework Security Failures (all fixed in this change)

| # | Severity | Issue | Status |
|---|---|---|---|
| 1 | **Critical** | Permission policies were never instantiated — `self.policies = []`, `_register_rules()` returned immediately. `FilesystemPolicy`/`TerminalPolicy` existed but were dead code, so all user-configured rules were silently ignored | Fixed |
| 2 | **Critical** | `.env` deny was bypassable via symlink (`notes.txt → .env` returned ALLOW) and entirely absent for shell commands (`cat .env` returned ALLOW) | Fixed |
| 3 | **High** | Subagents called `registry.execute()` directly, bypassing the permission gate completely | Fixed |
| 4 | **High** | Compaction re-injected attacker-influenced summaries as `role="system", pinned=True` — persistent prompt injection that survived truncation | Fixed |
| 5 | **Medium** | SSRF: `web` tool fetched any model-supplied URL, any scheme, `follow_redirects=True`, no host/IP validation → cloud metadata reachable | Fixed |
| 6 | **Medium** | Browser screenshot `output_path` unvalidated → arbitrary file overwrite | Fixed |
| 7 | **Medium** | Redaction missed Google/Firebase keys, most providers, JWTs, and many token formats; no log filter existed | Fixed |
| 8 | **Medium** | `firebase_auth.py` debug-logged raw HTTP request lines carrying `idToken`/`refreshToken` | Fixed |
| 9 | **Medium** | Secrets written world-readable: `~/.agent/agent.db` was `0644`; `write_text()`-then-`chmod()` left a race window on `~/.agent/env` | Fixed |
| 10 | **Medium** | MCP injected AWS credentials into any server whose name/args contained the substring `aws` | Fixed |
| 11 | **Low** | Postgres connection string passed as argv → visible in `ps` | Fixed |
| 12 | **Low** | AWS MCP preset pinned `@latest`, not a fixed version | Documented; left as-is (see §5) |

---

## 3. Dead LLM provider files removed

The LLM connection is established exclusively from the TUI
(`src/agent/tui/app.py`, `src/agent/tui/run.py` → `llm/runtime.py` →
`OpenAICompatibleProvider`). Four provider files were completely unreferenced —
no imports, no dynamic loading, no registry registration, no packaging or docs
reference, and zero test references:

| File | Lines | Class | Referenced? |
|---|---|---|---|
| `src/agent/llm/anthropic.py` | 50 | `AnthropicProvider` | No — dead |
| `src/agent/llm/groq.py` | 400 | `GroqProvider` | No — dead |
| `src/agent/llm/nvidia.py` | 690 | `NVIDIAProvider` | No — dead |
| `src/agent/llm/local.py` | 0 | — | No — empty file |

Removed. Retained because they are live:

| File | Why it stays |
|---|---|
| `llm/provider.py` | Core registry, imported across the framework |
| `llm/runtime.py` | Used by the TUI to install the provider |
| `llm/openai_compatible.py` | The provider actually instantiated |
| `llm/rate_limiter.py` | Used by `context/manager.py` |

Net: **1,140 lines of dead code removed**, and the file that carried the leaked
Groq key no longer exists.

---

## 4. Guards added to prevent recurrence

| Guard | File |
|---|---|
| gitleaks rules for every leaked key class | `.gitleaks.toml` |
| Dependency-free local pre-commit hook | `.githooks/pre-commit` |
| CI secret scan over **full history** + tracked-file assertions | `.github/workflows/secret-scan.yml` |

Enable the local hook:

```bash
git config core.hooksPath .githooks
```

---

## 5. Outstanding actions requiring operator access

These cannot be completed from the code and are **required** to close the
exposure:

1. **Revoke all three credentials** (§1). Revocation is the only action that
   actually stops use — history rewriting alone is insufficient.
2. **Restrict or replace the Firebase key**, and check project `cloudsever-526fb`
   for unauthorised access.
3. **Rewrite git history** to purge the credentials, then force-push and have
   all clones re-cloned:

   ```bash
   pip install git-filter-repo
   git filter-repo --invert-paths \
     --path 'assets/main project - Repository debugging analysis.html' \
     --path 'assets/main project - Repository debugging analysis.pdf'
   git push --force --mirror origin
   ```

   History rewriting is destructive and rewrites shared commit IDs, so it is
   deliberately **not** performed automatically. Treat any collaborator clone
   as compromised and re-clone it after the force-push.

4. **Pin the AWS MCP preset** from `@latest` to an exact version, so a future
   upstream release cannot change what receives live cloud credentials.

5. **Consider enabling Firestore/Auth security rules tests** — this repository
   is an agent framework with no Firebase ruleset of its own, but the leaked
   project is the one the sign-in flow targets.