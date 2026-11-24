import json
import os
import re
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.patient_retrieval_agent import retrieve_patient_records_for_agent
from services.patient_record_service import PatientRecordRetrievalError


PATIENTS_FILE = PROJECT_ROOT / "dataset" / "patients.json"
DATASET_DIR = PROJECT_ROOT / "dataset"
DATABASE_FILE = Path(
    os.environ.get(
        "PATIENT_REQUESTS_DB",
        str(PROJECT_ROOT / "backend" / "requests.sqlite3"),
    )
)
INDEX_FILE = PROJECT_ROOT / "backend" / "index.html"
PATIENT_ID_PATTERN = re.compile(r"SYN-\d{4}\Z")
MAX_BODY_SIZE = 4096
PORT = int(os.environ.get("PATIENT_REQUESTS_PORT", "8000"))


def load_patient_ids():
    with PATIENTS_FILE.open(encoding="utf-8") as handle:
        patients = json.load(handle)
    return {patient["patient_id"] for patient in patients}


def initialize_database(database_file):
    database_file.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_file)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                result_json TEXT,
                error TEXT
            )
            """
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(requests)")
        }
        if "result_json" not in columns:
            connection.execute("ALTER TABLE requests ADD COLUMN result_json TEXT")
        if "error" not in columns:
            connection.execute("ALTER TABLE requests ADD COLUMN error TEXT")


def create_agent_request(patient_id, patient_ids, database_file):
    if not isinstance(patient_id, str):
        raise ValueError("patient_id must be a string")

    patient_id = patient_id.strip()
    if not PATIENT_ID_PATTERN.fullmatch(patient_id):
        raise ValueError("Enter a patient ID in the format SYN-0001")
    if patient_id not in patient_ids:
        raise LookupError("Patient ID was not found")

    request = {
        "request_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with closing(sqlite3.connect(database_file)) as connection, connection:
        connection.execute(
            """
            INSERT INTO requests (request_id, patient_id, status, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                request["request_id"],
                request["patient_id"],
                request["status"],
                request["created_at"],
            ),
        )
    return request


def update_agent_request(request_id, status, database_file, result=None, error=None):
    result_json = json.dumps(result) if result is not None else None
    with closing(sqlite3.connect(database_file)) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE requests
            SET status = ?, result_json = ?, error = ?
            WHERE request_id = ?
            """,
            (status, result_json, error, request_id),
        )
        if cursor.rowcount != 1:
            raise LookupError(f"Request {request_id} was not found")


def get_agent_request(request_id, database_file):
    with closing(sqlite3.connect(database_file)) as connection, connection:
        row = connection.execute(
            """
            SELECT request_id, patient_id, status, created_at, result_json, error
            FROM requests
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
    if row is None:
        return None

    request = {
        "request_id": row[0],
        "patient_id": row[1],
        "status": row[2],
        "created_at": row[3],
    }
    if row[4] is not None:
        request.update(json.loads(row[4]))
    if row[5] is not None:
        request["error"] = row[5]
    return request


class RequestHandler(BaseHTTPRequestHandler):
    patient_ids = set()
    database_file = DATABASE_FILE
    dataset_dir = DATASET_DIR

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = INDEX_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        match = re.fullmatch(r"/api/requests/([0-9a-fA-F-]{36})", self.path)
        if match:
            request = get_agent_request(match.group(1), self.database_file)
            if request is not None:
                self._send_json(200, request)
                return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/requests":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return

        if content_length <= 0:
            self._send_json(400, {"error": "A JSON request body is required"})
            return
        if content_length > MAX_BODY_SIZE:
            self._send_json(413, {"error": "Request body is too large"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON"})
            return

        if not isinstance(payload, dict):
            self._send_json(400, {"error": "Request body must be a JSON object"})
            return

        try:
            request = create_agent_request(
                payload.get("patient_id"), self.patient_ids, self.database_file
            )
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        except LookupError as error:
            self._send_json(404, {"error": str(error)})
            return

        try:
            result = retrieve_patient_records_for_agent(
                request["patient_id"], self.dataset_dir
            )
        except PatientRecordRetrievalError as error:
            update_agent_request(
                request["request_id"],
                "failed",
                self.database_file,
                error=str(error),
            )
            self._send_json(
                500,
                {
                    **request,
                    "status": "failed",
                    "error": str(error),
                },
            )
            return

        update_agent_request(
            request["request_id"],
            "completed",
            self.database_file,
            result=result,
        )
        self._send_json(201, {**request, **result, "status": "completed"})

    def log_message(self, format_string, *args):
        return


def main():
    RequestHandler.patient_ids = load_patient_ids()
    initialize_database(DATABASE_FILE)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), RequestHandler)
    print(f"Patient request UI available at http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
