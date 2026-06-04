# Loan Default ML

## MLflow UI

From the project root (with the venv active):

```powershell
uv sync
python scripts/start_mlflow.py
```

Or on Windows:

```powershell
.\scripts\start_mlflow.ps1
```

Open **http://127.0.0.1:5000** in your browser.

If you upgraded MLflow and see a database schema error:

```powershell
python -m mlflow db upgrade sqlite:///mlflow.db
```

**Note:** PowerShell may show red text for MLflow log lines written to stderr (security middleware, Windows job warning). Those are informational; if you see `Uvicorn running on http://127.0.0.1:5000`, the UI is up. Prefer `python scripts/start_mlflow.py` over bare `mlflow ui` to avoid that noise.
