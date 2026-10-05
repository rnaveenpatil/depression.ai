# First-run login (TUI) + Firebase / Gmail sign-in

When `depression-tui` starts on a machine that has no stored profile, the TUI
pushes a **welcome page** on top of the normal app. Sign in with Google and the
page stores your name, email and photo on this machine — then it never appears
again. Skipping is always allowed and never blocks the CLI.

```
src/agent/tui/onboarding/
├── __init__.py          public surface
├── firebase_config.py   resolves the Firebase web config (env / json file)
├── firebase_auth.py     loopback callback server + Firebase REST calls
├── google_oauth.py      google-auth-oauthlib installed-app flow + requests
├── profile.py           user.json / user_tokens.json (the "new user?" gate)
├── identity.py          exports the profile into env + conversation context
└── welcome.py           WelcomeScreen (Textual ModalScreen) + the gate helper
```

The only hook into existing code is three lines in `src/agent/tui/app.py`
(`on_mount` → `call_later(self._maybe_onboard)`).

---

## 1. One-time Firebase setup (console)

1. <https://console.firebase.google.com> → **Add project**.
2. **Authentication** → *Get started* → enable the **Google** provider
   (pick a support email, Save).
3. **Project settings** → *Your apps* → **Web app** (`</>`) → register app.
4. Copy the `firebaseConfig` object. It looks like:

   ```js
   const firebaseConfig = {
     apiKey: "AIza…",
     authDomain: "your-project.firebaseapp.com",
     projectId: "your-project",
     appId: "1:1234567890:web:abc123",
     measurementId: "G-XXXXXXX",
   };
   ```

5. **Authentication → Settings → Authorized domains**: add
   `localhost` (already allowed by default) and any domain you use.

The web config is *public by design* — Firebase ships it inside every web app.
Never commit a service-account JSON key.

## 2. Point the CLI at that project

Either environment variables:

```bash
export FIREBASE_API_KEY="AIza…"
export FIREBASE_AUTH_DOMAIN="your-project.firebaseapp.com"
export FIREBASE_PROJECT_ID="your-project"
export FIREBASE_APP_ID="1:1234567890:web:abc123"     # optional
```

…or a file (handy for CI images and other users of the same machine):

```bash
mkdir -p ~/.config/depression
cat > ~/.config/depression/firebase.json <<'JSON'
{ "apiKey": "AIza…", "authDomain": "your-project.firebaseapp.com", "projectId": "your-project" }
JSON
```

The **whole Firebase Console > Project settings > General export can be pasted
in as-is** — `client: [{ … api_key: [{current_key}], oauth_client: [{…}] }]` is
understood, and the `client_type: 3` ("Web client") entry is what the OAuth flow
uses:

```json
{
  "project_id": "cloudsever-526fb",
  "project_number": "605436333256",
  "storage_bucket": "cloudsever-526fb.firebasestorage.app",
  "authDomain": "cloudsever-526fb.firebaseapp.com",
  "api_key": [{ "current_key": "AIza…" }],
  "oauth_client": [
    { "client_id": "…android….apps.googleusercontent.com", "client_type": 1 },
    { "client_id": "….apps.googleusercontent.com",           "client_type": 3 }
  ]
}
```

A project with **only an Android app** is enough for the OAuth flow — a Firebase
*Web app* (an `appId` containing `:web:`) is only required by the built-in
browser page. To unlock that too, register one:
Firebase Console > Project settings > **Add app > Web**, then add
`authDomain` and the new `appId` to this file.

Inline JSON also works: `DEPRESSION_FIREBASE_WEB_APP_CONFIG='{"apiKey":"…",…}'`,
and a path via `FIREBASE_CONFIG_FILE=/path/to/firebase.json`.

Finally, turn the provider on:
**Authentication > Sign-in method > Google > Enable**, with a support email.
Without that, Firebase answers `OPERATION_NOT_ALLOWED : The given sign-in
provider is disabled for this Firebase project.`

## 3. Two sign-in methods

### 3a. google-auth-oauthlib (preferred, when installed)

```bash
pip install google-auth-oauthlib requests
```

Then download an OAuth client file from
**Firebase Console > Project Settings > Your apps > your Web app >
Download and configure** and point at it:

```bash
export FIREBASE_GOOGLE_CLIENT_SECRETS=/path/to/client_secrets.json
# or: ~/.config/depression/client_secrets.json, or ./client_secrets.json
```

Pressing `enter`/`g` then runs Google's official installed-app flow —
`InstalledAppFlow.run_local_server(port=0)` starts its own loopback server,
opens the browser and blocks until consent, so **nothing is copied out of the
browser window**:

```python
from google_auth_oauthlib.flow import InstalledAppFlow

flow = InstalledAppFlow.from_client_secrets_file(
    "client_secrets.json",
    scopes=["openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile"],
)
credentials = flow.run_local_server(port=0)
google_id_token = credentials.id_token
```

That Google token is exchanged for a Firebase session, exactly like the page
below:

```
POST https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp?key=$FIREBASE_API_KEY
{"postBody": "id_token=<google_id_token>&providerId=google.com",
 "requestUri": "http://localhost", "returnSecureToken": true}
```

The returned `idToken` is verified through `accounts:lookup`, so both methods
end up with the same authoritative user record. If the OAuth step fails (wrong
redirect URI, consent denied, …) the CLI silently falls back to the browser
page below. `WelcomeScreen(enable_oauth=False)` disables the OAuth path.

### 3b. Built-in browser page (no extra dependencies)

## 4. What happens on first launch

1. `DepressionApp.on_mount` schedules `_maybe_onboard`.
2. `should_onboard()` checks, in order:
   * `DEPRESSION_ONBOARDING` is not `off/0/no/false`
   * stdin **and** stdout are a TTY (pipes and `depression "query"` stay silent)
   * the app is not running headless (test harnesses are skipped)
   * no usable `user.json` exists yet
3. `WelcomeScreen` is pushed. Keys: `enter`/`g` sign in, `t` paste an ID token,
   `c` continue as guest, `esc` skip. A countdown (default 90s) auto-continues
   so a first launch can never hang.
4. **Sign in with Google** starts a one-shot HTTP server on `localhost:<random>`
   and opens your browser. The host must be the name `localhost` rather than
   `127.0.0.1`, because Firebase matches the page origin against the project's
   authorized-domain list and that list does not alias the literal IP — with
   `127.0.0.1` the popup stalls on `__/auth/handler` and never reaches Google's
   login page. Google shows the normal consent screen; the page posts the
   Firebase `idToken` back to the loopback URL; the CLI verifies it with
   `identitytoolkit/v1/accounts:lookup` and saves the profile.
5. Stored files (both `0600`, next to `agent.db`):

   | file | contents |
   |------|----------|
   | `user.json` | uid, name, email, photo, provider, `created_at`, `last_login_at`, `login_count` |
   | `user_tokens.json` | Firebase `id_token`, `refresh_token`, expiry |

   | platform | path |
   |----------|------|
   | Linux | `~/.local/share/depression/` |
   | macOS | `~/Library/Application Support/depression/` |
   | Windows | `%APPDATA%\depression\` |

## 5. What the agent gets afterwards

The page is asked **only on the very first launch**. From the second launch
onwards the stored profile is silently re-exported, exactly like the other
credentials in this app:

* written to the global env file `~/.agent/env` (0600 — same file as the AWS
  keys and provider API keys), and
* mirrored into `os.environ`, so `bash`, MCP servers and any subprocess the
  agent spawns inherit them, and
* injected once per session as a pinned system message, so the model itself
  knows who it is talking to.

| env var | value |
|---------|-------|
| `DEPRESSION_USER_NAME` | display name from Gmail |
| `DEPRESSION_USER_EMAIL` | Gmail address |
| `DEPRESSION_USER_UID` | Firebase uid |
| `DEPRESSION_USER_PHOTO` | photo URL (empty if Google gave none) |
| `DEPRESSION_USER_PROVIDER` | `google.com` |
| `DEPRESSION_USER_EMAIL_VERIFIED` | `true` / `false` |
| `DEPRESSION_USER_SIGNED_IN` | `true` after Google login, `false` for guest |
| `DEPRESSION_USER_LAST_LOGIN` | unix timestamp of the last sign-in |

Inside the agent:

```bash
echo "$DEPRESSION_USER_NAME"   # visible to any tool that reads the environment
```

The Firebase refresh token is intentionally **not** exported; it stays in the
0600 `user_tokens.json` and is only used through
`FirebaseGoogleAuth.refresh_tokens`.

```python
from agent.tui.onboarding.identity import clear_identity, load_identity
load_identity()      # -> {DEPRESSION_USER_EMAIL: 'patil@gmail.com', ...}
clear_identity()     # sign-out: removes every var from env + ~/.agent/env
```

## 6. Headless machines

There is no browser, so press `t` on the welcome page and paste a Firebase ID
token instead — it is verified through the same REST call. Or skip login and
use `c` (guest), which stores a local-only profile.

## 7. Environment switches

| variable | effect |
|----------|--------|
| `DEPRESSION_ONBOARDING=off\|0\|no\|false` | never show the page |
| `DEPRESSION_ONBOARDING=force` | show it again even when signed in |
| `DEPRESSION_ONBOARDING_TIMEOUT` | browser sign-in timeout, default `180`s |
| `DEPRESSION_ONBOARDING_SKIP_TIMEOUT` | auto-continue countdown, `0` disables |
| `FIREBASE_GOOGLE_CLIENT_SECRETS` | `client_secrets.json` for the OAuth flow |
| `FIREBASE_OAUTH_CLIENT_ID` | OAuth client id when `firebase.json` has none |
| `FIREBASE_OAUTH_CLIENT_SECRET` | only needed by web clients that require it |

## 8. Signing out / resetting

```bash
rm ~/.local/share/depression/user.json ~/.local/share/depression/user_tokens.json
# or, from Python (removes the exported env vars too):
#   from agent.tui.onboarding.identity import sign_out; sign_out()
```

Next launch shows the welcome page again.

## 9. Tests

```bash
python -m pytest tests/test_onboarding.py -q
```

They never touch the network (`httpx.MockTransport`) and never write outside
`tmp_path`.