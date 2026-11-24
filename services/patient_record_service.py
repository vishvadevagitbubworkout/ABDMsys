import json
import re
from pathlib import Path

from services.dataset_registry import (
    CATEGORY_REGISTRY,
    HOSPITALS,
    PATIENTS_FILENAME,
    category_file,
)
from services.record_validators import inspect_dataset, read_category_records


DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
PATIENT_ID_PATTERN = re.compile(r"SYN-\d{4}\Z")


class PatientRecordRetrievalError(Exception):
    """Raised when patient records cannot be safely retrieved."""


class PatientNotFoundError(PatientRecordRetrievalError):
    """Raised when the requested patient is absent from the dataset."""


def retrieve_patient_records(patient_id, dataset_dir=DEFAULT_DATASET_DIR):
    if not isinstance(patient_id, str) or not PATIENT_ID_PATTERN.fullmatch(patient_id):
        raise PatientRecordRetrievalError("Enter a patient ID in the format SYN-0001")

    dataset_dir = Path(dataset_dir)
    patients_file = dataset_dir / PATIENTS_FILENAME
    try:
        with patients_file.open(encoding="utf-8") as handle:
            patients = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PatientRecordRetrievalError(
            f"Unable to read patient profiles from {patients_file}: {error}"
        ) from error

    if not isinstance(patients, list):
        raise PatientRecordRetrievalError(
            f"Patient profiles in {patients_file} must be a JSON list"
        )
    patient = next(
        (
            item
            for item in patients
            if isinstance(item, dict) and item.get("patient_id") == patient_id
        ),
        None,
    )
    if patient is None:
        raise PatientNotFoundError(f"Patient ID {patient_id} was not found")

    validation = inspect_dataset(
        dataset_dir,
        expected_patient_count=None,
        expected_visit_count=None,
    )
    if not validation["valid"]:
        issues = (
            validation["invalid_records"]
            + validation["missing_references"]
            + validation["duplicate_ids"]
        )
        details = "; ".join(_format_validation_issue(issue) for issue in issues[:5])
        remaining = len(issues) - 5
        if remaining > 0:
            details += f"; and {remaining} more validation issue(s)"
        raise PatientRecordRetrievalError(
            f"Dataset validation failed: {details}"
        )

    hospitals = {
        hospital: {category: [] for category in CATEGORY_REGISTRY}
        for hospital in HOSPITALS
    }
    for hospital in HOSPITALS:
        for category in CATEGORY_REGISTRY:
            path = category_file(dataset_dir, hospital, category)
            try:
                records = read_category_records(path, category)
            except ValueError as error:
                raise PatientRecordRetrievalError(str(error)) from error
            hospitals[hospital][category] = [
                dict(record)
                for record in records
                if record.get("patient_id") == patient_id
            ]

    return {
        "patient_id": patient_id,
        "patient": dict(patient),
        "hospitals": hospitals,
    }


def _format_validation_issue(issue):
    if "error" in issue:
        message = issue["error"]
    elif "duplicate_of" in issue:
        message = f"duplicate {issue['category']} ID {issue['record_id']}"
    else:
        message = f"missing {issue['reference']} reference {issue.get('value')}"
    return f"{issue['source']}: {message}"
