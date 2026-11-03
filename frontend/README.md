# RoleSync: frontend

A step-by-step web app on top of the FastAPI backend: upload a resume, confirm the profile, paste a job,
check the fit, review and approve the tailored resume, choose an ATS-safe design, then write, review and send
the e-mail with the tailored PDF attached.

## Run it

```bash
# 1. backend (from the project root)
.venv/bin/python -m uvicorn app.main:app --port 8000

# 2. frontend (from frontend/)
npm install
npm run dev          # http://127.0.0.1:5173
```

The browser only talks to `/api`; Vite forwards that to the backend, so there is no CORS setup and no backend
address in the code. To use another backend: `VITE_BACKEND_URL=http://host:port npm run dev` (see `.env.example`).
Ollama must be running for the fit review, tailoring and e-mail steps (they can take a minute or two).

## Checks

```bash
npm test             # unit and screen tests (vitest)
npm run typecheck    # strict TypeScript
npm run lint
npm run build
```

## Layout

- `src/api.ts` typed client; turns API errors into readable messages
- `src/flowModel.ts`, `src/flow.tsx` the user's progress (kept in `sessionStorage`, so a reload resumes)
- `src/steps.ts` which step unlocks when
- `src/steps/*` one screen per step
