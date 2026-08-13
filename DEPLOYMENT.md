# Deployment

Two parts: the FastAPI backend (`api/`) and the React frontend (`frontend-react/`).
They're separate processes in dev; in production the built React app is served as
static files by the same FastAPI process (one process to run, one port to expose).

## 1. Local development (what's running right now)

Two terminals, from the project root:

```bash
# terminal 1 — backend, auto-reloads on code changes
cd aircraft-rul-prognostics
uv sync
uv run uvicorn api.main:app --reload --port 8000

# terminal 2 — frontend, auto-reloads on code changes, proxies /api/* to :8000
cd aircraft-rul-prognostics/frontend-react
npm install
npm run dev -- --port 5173
```

Open `http://localhost:5173/`. `vite.config.ts` proxies `/api` to `http://localhost:8000`,
so the two processes talk to each other with no CORS config needed. `http://localhost:8000/docs`
gives you the interactive API docs directly.

On Windows machines without Node.js and without `winget` (Group Policy can block it),
use a portable install — no admin rights needed:

```powershell
Invoke-WebRequest https://nodejs.org/dist/v22.11.0/node-v22.11.0-win-x64.zip -OutFile node.zip
Expand-Archive node.zip -DestinationPath node-extract
# add node-extract/node-v22.11.0-win-x64/ to PATH for the session, or call node.exe/npm.cmd directly
```

If `npm run build` or `npm run dev` fails with `Cannot find native binding` for
`@rolldown/binding-win32-x64-msvc`, run `npm install @rolldown/binding-win32-x64-msvc --no-save`
once — an npm optional-dependency bug on Windows sometimes skips it.

## 2. Single-process production deploy

Build the React app, then point FastAPI's static mount at the build output instead of
the old `frontend/` dashboard.

```bash
cd frontend-react
npm run build          # outputs frontend-react/dist/
```

`api/main.py` currently ends with:

```python
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "frontend"), html=True), name="frontend")
```

For production, change that one line to point at `frontend-react/dist` instead:

```python
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "frontend-react" / "dist"), html=True), name="frontend")
```

(Not changed yet in this repo — both frontends still exist side by side; see
`HANDOVER.md` for why. Flip this line when you're ready to cut over.)

Then run one process:

```bash
uv sync
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

No `--reload` in production (it watches files and restarts on change — wasteful and
occasionally flaky under load). One port (`8000`) serves both the API (`/api/*`) and
the built frontend (everything else).

## 3. Environment / process notes

- **No `.env` or secrets today** — the app has no auth and no external service
  credentials. If you add any (a database, an auth provider), keep secrets out of
  git and load them via environment variables, not hardcoded config.
- **No process manager configured yet.** For a long-running deploy, wrap the
  `uvicorn` command in whatever your host already uses:
  - Linux: a `systemd` unit (`ExecStart=uv run uvicorn api.main:app --host 0.0.0.0 --port 8000`)
    or a supervisor like `pm2`/`supervisord`.
  - Windows: `nssm` to run it as a service, or Task Scheduler with "restart on failure."
  - Any container host: see below.
- **No Docker packaging yet** (tracked as a known gap in `HANDOVER.md`). A minimal
  `Dockerfile` for this app would be:
  ```dockerfile
  FROM python:3.11-slim
  RUN pip install uv
  WORKDIR /app
  COPY . .
  RUN cd frontend-react && npm ci && npm run build   # needs a Node build stage or multi-stage build
  RUN uv sync --frozen
  # then flip api/main.py's static mount to frontend-react/dist as in step 2
  EXPOSE 8000
  CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
  This needs Node available in the image (or a proper multi-stage build with a
  `node:slim` stage for `npm run build`, copying just `dist/` into the final Python
  image) — worth doing properly rather than pasting this as-is when you actually
  containerize.
- **Model artifacts are loaded from disk at startup** (`models/`, `artifacts/`) —
  whatever host you deploy to needs those directories present; they're not
  regenerated at runtime. If you retrain, redeploy with the new `.joblib`/`.pt` files
  in place.
- **CORS is wide open** (`allow_origins=["*"]`) — fine when the frontend is served by
  the same FastAPI process (same origin, so CORS doesn't even apply), but tighten this
  to specific origins if the frontend and API ever run on different hosts/domains.

## 4. What's deliberately not covered here

HTTPS/TLS termination, a reverse proxy (nginx/Caddy) in front of uvicorn, horizontal
scaling, and CI/CD — none of that exists yet and none was asked for. `HANDOVER.md` has
the full list of known gaps (auth, tests, model versioning, retraining pipeline).
