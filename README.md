# Bharat Health Agent

## Requirements

- Python 3
- A compatible patient dataset in `dataset/`

No third-party Python packages are required.

This repository intentionally excludes generated/synthetic data, data-generation
scripts, validation reports, tests, backups, and local SQLite state.

## Dataset layout

Place the dataset outside Git control under `dataset/`, using this structure:

```text
dataset/
  patients.json
  Hospital_A/
  Hospital_B/
  Hospital_C/
```

Each hospital must provide the same record-category directories defined by
`services/dataset_registry.py`. Patient profiles and records must use the
`patient_id` and hospital fields expected by the retrieval service.

## Run

From the project directory:

```powershell
python backend\app.py
```

Open <http://127.0.0.1:8000>.

To use another port:

```powershell
$env:PATIENT_REQUESTS_PORT = "8001"
python backend\app.py
```

## API

Create a retrieval request:

```text
POST /api/requests
Content-Type: application/json

{"patient_id": "PATIENT-ID"}
```

Fetch a request:

```text
GET /api/requests/{request_id}
```

Requests and results are stored locally in `backend/requests.sqlite3`. This file
is excluded from Git and is created automatically when the API starts.
