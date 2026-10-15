# IRM Dashboard Backend and Frontend for the files  

This FastAPI service provides read-only transaction data, system capability
reporting, and an API contract for the separately developed frontend. It does
not change account balances or transaction details. The simulation endpoint
currently returns `501 NOT_IMPLEMENTED`: the existing C++ executable only prints
a startup message and has no simulation CLI or structured output.

## Windows PowerShell setup

From the repository root:

```powershell
Set-Location dashboard/backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Fill in `DB_USER` and `DB_PASSWORD` in `.env`. `DB_HOST`, `DB_PORT`, and
`DB_NAME` default to `localhost`, `3306`, and `resource_management_db`. The
example file contains placeholders only. Keep real credentials in the ignored
`.env` file or set them in the current PowerShell session.

If PowerShell blocks virtual environment activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal,
then activate again. This changes policy only for the current process.

To use current-session environment variables instead of `.env`:

```powershell
$env:DB_HOST = 'localhost'
$env:DB_PORT = '3306'
$env:DB_NAME = 'resource_management_db'
$env:DB_USER = Read-Host 'MySQL user'
$securePassword = Read-Host 'MySQL password' -AsSecureString
$env:DB_PASSWORD = [System.Net.NetworkCredential]::new('', $securePassword).Password
```

The backend also accepts `ALLOWED_ORIGINS`, `CPP_ENGINE_PATH`, and
`SIMULATION_TIMEOUT_SECONDS`. Their defaults allow the local Vite origins,
point to `core_engine/build/IntelligentResourceManager.exe`, and use a 30-second
simulation timeout.

## Run, test, and API docs

From `dashboard/backend` with the virtual environment active:

```powershell
py -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> for interactive API documentation. From the
repository root, run the tests:

```powershell
py -m pytest dashboard/backend/tests -v
```

If C++ needs rebuilding, from the repository root use the project CMake build:

```powershell
cmake -S core_engine -B core_engine/build -DMYSQL_SERVER_ROOT="C:/Program Files/MySQL/MySQL Server 8.0"
cmake --build core_engine/build --target IntelligentResourceManager
```

The existing executable is not currently capable of running a simulation from
the API. A minimal C++ CLI that accepts algorithm options and emits stable JSON
would be needed before enabling simulation calls. The backend does not guess at
or parse human-readable startup output.

## Data and capability notes

Read endpoints query the existing `resource_management_db.Transactions` table
with parameterized SQL. `Amount` is returned as a decimal string to preserve
precision, and timestamp columns are ISO-8601 strings or `null`. The schema and
seed files are not modified. API limitations and the complete response contract
are documented in `../../docs/api_contract.md`.
