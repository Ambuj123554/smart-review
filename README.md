# Smart Review Assistant

An AI-powered automated pull request reviewer. It listens for GitHub webhook
events on a repository, fetches the changed files of a pull request, has an
LLM (Google Gemini) review the added lines with strict anti-hallucination
validation, and posts a single consolidated inline review back to the PR.

Target users: small student and open-source teams with no dedicated reviewer.

## What it does

1. **Webhook receiver** — `POST /webhook/github` accepts GitHub
   `pull_request` events (`opened`, `synchronize`, `reopened`) and verifies
   the `X-Hub-Signature-256` HMAC signature.
2. **Diff fetching** — `GET /repos/{owner}/{repo}/pulls/{n}/files` (paginated,
   capped by `MAX_FILES_PER_PR` / `GITHUB_MAX_LINES`) parsed into per-file
   added lines on the new side of the diff. Binary files, deleted files and
   files without patches are skipped.
3. **LLM review** — each file's added lines are sent to the configured
   provider (Gemini by default, or Groq via `LLM_PROVIDER=groq`) with a
   prompt that (a) marks the diff content as untrusted input, (b) demands a
   strict JSON array of findings and (c) requires line numbers from the
   added-lines list. Findings whose line is not actually an added line are
   dropped (hallucination guard).
4. **Batched review posting** — all findings are submitted in ONE
   `POST /repos/{owner}/{repo}/pulls/{n}/reviews` call with inline comments
   (`path`, `line`, `side=RIGHT`, `commit_id=head_sha`). Duplicate comments
   from previous pushes are detected and skipped. Rate limits and transient
   failures are retried with exponential backoff.
5. **History & analytics** — every PR and finding is stored in SQLite
   (`reviews.db`); `GET /stats` aggregates counts by severity, category and
   repository.

## Architecture

```
GitHub ──webhook──►  app/main.py (FastAPI)
                        │
                        ├── app/github/webhooks.py    verify + filter + background task
                        │        └── _process_review()
                        │              ├── app/github/client.py    paginated /pulls/{n}/files
                        │              ├── app/github/diff.py      unified-diff parser
                        │              ├── app/ai/diff_reviewer.py Gemini review (validated)
                        │              │      └── app/ai/guardrails.py   fence neutralization
                        │              ├── app/github/comments.py  batched review POST
                        │              │        (dedupe + retries)
                        │              └── app/database/db.py      persist PR + findings
                        │
                        ├── POST /review        ad-hoc snippet review (AST + Gemini)
                        ├── GET  /reviews       snippet review history
                        ├── GET  /stats         PR finding analytics
                        └── GET  /health
```

## Setup

### 1. Requirements

- Python 3.10+
- A GitHub account that can create Apps or tokens
- A Gemini API key (free tier works)

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Purpose |
|---|---|---|
| `GITHUB_WEBHOOK_SECRET` | yes | HMAC secret shared with the GitHub webhook |
| `GITHUB_TOKEN` | yes* | Personal Access Token for reading PRs and posting reviews |
| `GEMINI_API_KEY` | yes* | Google Gemini API key (*required when `LLM_PROVIDER=gemini`) |
| `GEMINI_MODEL` | no | Model id, defaults to `gemini-2.0-flash` |
| `LLM_PROVIDER` | no | LLM backend: `gemini` (default) or `groq` |
| `GROQ_API_KEY` | only for groq | Groq API key (`gsk_...`) |
| `GROQ_MODEL` | no | Groq model id, defaults to `llama-3.3-70b-versatile` |
| `MAX_FILES_PER_PR` | no | Max files per PR analysed (default 20) |
| `GITHUB_MAX_LINES` | no | Max added lines per PR analysed (default 3000) |
| `DATABASE_PATH` | no | SQLite file (default `reviews.db`) |
| `HOST` / `PORT` / `DEBUG` | no | Server settings |

*A GitHub App installation token flow is planned; for now a classic PAT
with **Contents: Read** and **Pull requests: Read & write** (nothing else —
least privilege) is enough.

### 3. Creating the token & webhook

1. GitHub → Settings → Developer settings → **Personal access tokens
   (classic)** → generate with only `repo:pull_requests` scope.
2. In the target repo → Settings → Webhooks → **Add webhook**:
   - Payload URL: `https://your-domain/webhook/github`
   - Content type: `application/json`
   - Secret: the same value as `GITHUB_WEBHOOK_SECRET`
   - Events: "Pull requests" only
3. Local testing with **ngrok**:
   ```bash
   ngrok http 8000
   ```
   Use the printed `https://...ngrok-free.app` URL as the Payload URL, and
   set `GITHUB_WEBHOOK_SECRET` in both `.env` and the webhook form.

### 4. Run

```bash
uvicorn app.main:app --reload --port 8000
```

Try the API docs at <http://127.0.0.1:8000/docs>.

## Docker

```bash
docker build -t smart-review-assistant .
docker run --rm --env-file .env -p 8000:8000 smart-review-assistant
```

## Running tests

```bash
pytest
```

Tests use an isolated temp SQLite database and fake/mock GitHub payloads —
no real GitHub or Gemini credentials are required.

## Testing with a real repo

The unit suite never talks to GitHub or Gemini. To exercise the full loop
against a real repository:

1. Create a throwaway repository (or use a fork) and enable **Issues**.
2. Fill in `.env`: `GITHUB_WEBHOOK_SECRET` (any long random string),
   `GITHUB_TOKEN` (a classic PAT with `repo:pull_requests`, contents read —
   see above), and `GEMINI_API_KEY`.
3. Start the app and a public tunnel:
   ```bash
   uvicorn app.main:app --port 8000
   ngrok http 8000
   ```
4. Add a webhook in the repo settings: payload URL
   `https://<ngrok-url>/webhook/github`, content type `application/json`,
   secret = your `GITHUB_WEBHOOK_SECRET`, events = **Pull requests** only.
5. Open a PR that changes a Python file with an obvious issue, e.g.:
   ```python
   def greet(names=[]):
       eval("names")
   ```
6. Expect, on the PR:
   - inline comments on the changed lines (the eval + mutable-default rules
     fire deterministically even without a Gemini key),
   - one **summary comment** with files analysed, severity/category counts
     and the overall risk level,
   - a `GET /stats` entry for your repo.
7. Push an empty follow-up commit (`synchronize`): the same findings should
   **not** be re-posted (dedupe on file + line + message).

Troubleshooting: a `401` from the webhook = secret mismatch;
`gemini_api_configured: false` in `GET /health` = key missing (AST findings
still work); no comments at all = check the app log for `GitHubApiError`
(token scope/repo name) or truncation warnings mentioning
`MAX_FILES_PER_PR`.

## Known limitations

- **No persistent queue** — reviews run as in-process FastAPI
  `BackgroundTasks`; a restart between webhook receipt and review completion
  loses that review silently (no retry, no dead-letter). A durable queue
  (Celery/arq or a DB-backed pending table) is the fix for production.
- **Rate limits** — individual 403/429 responses are retried with
  `Retry-After`/backoff, but there is no global request budget, no secondary
  rate-limit awareness, and each analysed file costs an extra
  content-fetch API call. A burst of PRs on a low-limit PAT can still exhaust
  quota until the window resets.
- **Model id unverified** — `GEMINI_MODEL` defaults to `gemini-2.0-flash`
  and `GROQ_MODEL` defaults to `llama-3.3-70b-versatile`; no test or CI run
  ever calls either API (no credentials), so both ids are assumed valid.
  Override the env vars if your account uses different ones; an invalid id
  is logged clearly at startup of the first review (`check GROQ_MODEL` for
  Groq).
- **AST analysis is Python-only** — non-`.py` files get LLM review of their
  added lines but are skipped for content fetch/AST rules.

## CI

`.github/workflows/ci.yml` runs `pytest` on every push to keep the suite
green on all branches.
