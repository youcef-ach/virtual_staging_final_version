# Operator Dashboard

React admin tool for the offline pipeline. Current scope: **Gate 1 pose-QA
review** — summary stats, filterable sweep queue (flagged / spot-check / all,
worst-first), per-sweep face scores with no-signal faces, and the labeled grid
composite. Pose micro-adjust (F6.3) and Gate 2 mask fixing come later.

## Run

```powershell
# stage the QA data (git-ignored):
New-Item -ItemType Directory -Force dashboard\public | Out-Null
Copy-Item pipeline\qa_output dashboard\public\qa -Recurse -Force

cd dashboard
npm install
npm run dev    # http://localhost:5174
```
