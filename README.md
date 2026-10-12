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
   capped by `GITHUB_MAX_FILES` / `GITHUB_MAX_LINES`) parsed into per-file
   added lines on the new side of the diff. Binary files, deleted files and
   files without patches are skipped.
3. **LLM review** — each file's added lines are sent to Gemini with a
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
| `GEMINI_API_KEY` | yes | Google Gemini API key |
| `GEMINI_MODEL` | no | Model id, defaults to `gemini-2.0-flash` |
| `GITHUB_MAX_FILES` | no | Max files per PR analysed (default 50) |
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

## CI

`.github/workflows/ci.yml` runs `pytest` on every push to keep the suite
green on all branches.
