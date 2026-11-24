# Bharat Health Agent

## Requirements

- Python 3
- A compatible patient dataset in `dataset/`
- Ollama running locally with `rafw007/qwen35-claude-coder:4b` available

No third-party Python packages are required. The summary is generated and
cross-checked locally; patient records are not sent to an external service.

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

## Local chronological summaries

The UI retrieves the patient's records, asks the local Ollama model to create
a dated timeline by selecting relevant source fields, then asks it to check
those fields against their cited records. The application verifies each
selected field exists in the source and renders its stored value directly,
rather than displaying model-written clinical claims. It also rejects invalid
citations and events grouped across different dates. Only the summary, source
IDs, and record counts are returned or newly saved; raw records are not
displayed in the UI or included in new API responses.

The default model is `rafw007/qwen35-claude-coder:4b`. If needed, install it
with Ollama and start the local Ollama service:

```powershell
ollama pull rafw007/qwen35-claude-coder:4b
```

The first request can take several minutes while Ollama loads the model.

To select another model already installed in Ollama:

```powershell
$env:PATIENT_SUMMARY_MODEL = "your-model:tag"
```

Summaries are AI-generated and are not clinical advice. The fact-check uses the
same local model, so it is a grounding aid rather than independent medical
verification. Existing request history from earlier versions may still contain
raw records in the local SQLite database.

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

Requests and summaries are stored locally in `backend/requests.sqlite3`. This
file is excluded from Git and is created automatically when the API starts.
