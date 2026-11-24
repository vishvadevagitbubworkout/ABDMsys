import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from services.dataset_registry import (
    CATEGORY_REGISTRY,
    EXPECTED_PATIENT_COUNT,
    EXPECTED_VISIT_COUNT,
    HOSPITALS,
    PATIENTS_FILENAME,
    PRIMARY_HOSPITAL_SHARE,
    category_file,
)


PATIENT_FIELDS = (
    "patient_id",
    "name",
    "age",
    "sex",
    "blood_group",
    "primary_hospital",
    "synthetic",
)
PATIENT_ID_PATTERN = re.compile(r"SYN-\d{4}\Z")
VISIT_ID_PATTERN = re.compile(r"VIS-\d{4}\Z")
EMPTY_ALLOWED_FIELDS = {
    "medications": {"end_date"},
    "lab_results": {"result_value", "unit"},
    "clinical_notes": {"discharge_summary"},
    "vital_signs": {
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
        "temperature_c",
        "respiratory_rate",
        "spo2",
        "weight_kg",
        "height_cm",
    },
}
VITAL_RANGES = {
    "systolic_bp": (50, 260),
    "diastolic_bp": (30, 150),
    "heart_rate": (20, 250),
    "temperature_c": (25, 45),
    "respiratory_rate": (4, 70),
    "spo2": (50, 100),
    "weight_kg": (1, 400),
    "height_cm": (30, 250),
}
VALID_VISIT_STATUSES = {"Completed", "Discharged", "Admitted", "Observation"}
VALID_MEDICATION_STATUSES = {"active", "discontinued", "completed"}
VALID_RESULT_STATUSES = {"final", "pending"}
VALID_ALLERGY_STATUSES = {"active", "resolved"}
VALID_PROCEDURE_STATUSES = {"completed", "scheduled"}


class DatasetValidationError(ValueError):
    def __init__(self, report):
        self.report = report
        summary = "; ".join(
            issue["error"] for issue in report["invalid_records"][:5]
        )
        if not summary:
            summary = "Dataset validation failed"
        super().__init__(summary)


def read_category_records(path, category):
    path = Path(path)
    config = CATEGORY_REGISTRY[category]
    try:
        if config["format"] == "csv":
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, strict=True)
                if reader.fieldnames != list(config["fields"]):
                    raise ValueError(
                        f"expected columns {list(config['fields'])}, "
                        f"found {reader.fieldnames}"
                    )
                records = []
                for row_number, row in enumerate(reader, start=2):
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"malformed CSV row {row_number}")
                    records.append(dict(row))
                return records

        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError("JSON root must be a list of records")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"JSON record {index + 1} must be an object")
            if set(record) != set(config["fields"]):
                raise ValueError(
                    f"JSON record {index + 1} has unexpected fields"
                )
        return records
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def inspect_dataset(
    dataset_dir,
    expected_patient_count=EXPECTED_PATIENT_COUNT,
    expected_visit_count=EXPECTED_VISIT_COUNT,
):
    dataset_dir = Path(dataset_dir)
    invalid_records = []
    duplicate_ids = []
    missing_references = []

    def invalid(source, error, record_id=None):
        issue = {"source": str(source), "error": str(error)}
        if record_id is not None:
            issue["record_id"] = record_id
        invalid_records.append(issue)

    patients_file = dataset_dir / PATIENTS_FILENAME
    try:
        with patients_file.open(encoding="utf-8") as handle:
            patients = json.load(handle)
        if not isinstance(patients, list):
            raise ValueError("patient JSON root must be a list")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        patients = []
        invalid(patients_file, error)

    if expected_patient_count is not None and len(patients) != expected_patient_count:
        invalid(
            patients_file,
            f"expected {expected_patient_count} patients, found {len(patients)}",
        )

    patient_by_id = {}
    for index, patient in enumerate(patients, start=1):
        source = f"{patients_file} record {index}"
        if not isinstance(patient, dict):
            invalid(source, "patient record must be an object")
            continue
        missing_fields = set(PATIENT_FIELDS).difference(patient)
        if missing_fields:
            invalid(source, f"missing patient fields: {sorted(missing_fields)}")
            continue
        patient_id = patient.get("patient_id")
        if not isinstance(patient_id, str) or not PATIENT_ID_PATTERN.fullmatch(patient_id):
            invalid(source, "invalid patient_id", patient_id)
        if isinstance(patient_id, str) and patient_id in patient_by_id:
            duplicate_ids.append({
                "category": "patients",
                "record_id": patient_id,
                "source": source,
            })
        elif isinstance(patient_id, str):
            patient_by_id[patient_id] = patient
        if patient.get("primary_hospital") not in HOSPITALS:
            invalid(source, "invalid primary_hospital", patient_id)
        if patient.get("synthetic") is not True:
            invalid(source, "patient is not marked synthetic", patient_id)
        try:
            age = int(patient.get("age"))
            if not 0 <= age <= 120:
                raise ValueError("age must be between 0 and 120")
        except (TypeError, ValueError) as error:
            invalid(source, error, patient_id)

    records_by_hospital = {
        hospital: {category: [] for category in CATEGORY_REGISTRY}
        for hospital in HOSPITALS
    }
    identifiers = {category: {} for category in CATEGORY_REGISTRY}
    for hospital in HOSPITALS:
        hospital_dir = dataset_dir / hospital
        if not hospital_dir.is_dir():
            invalid(hospital_dir, "hospital directory is missing")
        for category, config in CATEGORY_REGISTRY.items():
            path = category_file(dataset_dir, hospital, category)
            if not path.is_file():
                invalid(path, "category file is missing")
                continue
            try:
                records = read_category_records(path, category)
            except ValueError as error:
                invalid(path, error)
                continue

            records_by_hospital[hospital][category] = records
            for index, record in enumerate(records, start=1):
                record_id = record.get(config["id_field"])
                source = f"{path} record {index}"
                if not isinstance(record_id, str) or not record_id.strip():
                    invalid(source, "record ID is missing")
                elif record_id in identifiers[category]:
                    duplicate_ids.append({
                        "category": category,
                        "record_id": record_id,
                        "source": source,
                        "duplicate_of": identifiers[category][record_id],
                    })
                else:
                    identifiers[category][record_id] = source

                patient_id = record.get("patient_id")
                if not isinstance(patient_id, str) or patient_id not in patient_by_id:
                    missing_references.append({
                        "source": source,
                        "record_id": record_id,
                        "reference": "patient_id",
                        "value": patient_id,
                    })
                if record.get("hospital") != hospital:
                    invalid(source, f"record hospital does not match {hospital}", record_id)
                if not _is_synthetic(record.get("synthetic")):
                    invalid(source, "record is not marked synthetic", record_id)

                for field in config["fields"]:
                    value = record.get(field)
                    if value is None:
                        invalid(source, f"field {field} is missing", record_id)
                    elif value == "" and field not in EMPTY_ALLOWED_FIELDS.get(category, set()):
                        invalid(source, f"field {field} is empty", record_id)

                date_value = record.get(config["date_field"])
                if date_value:
                    try:
                        _parse_timestamp(date_value)
                    except (TypeError, ValueError) as error:
                        invalid(source, f"invalid {config['date_field']}: {error}", record_id)

                _validate_category_values(
                    category, record, source, record_id, invalid
                )

    visits_by_id = {}
    for hospital in HOSPITALS:
        for visit in records_by_hospital[hospital]["visits"]:
            visit_id = visit.get("visit_id")
            if not isinstance(visit_id, str) or visit_id in visits_by_id:
                continue
            visits_by_id[visit_id] = visit
            if not isinstance(visit_id, str) or not VISIT_ID_PATTERN.fullmatch(visit_id):
                invalid(
                    category_file(dataset_dir, hospital, "visits"),
                    "invalid visit_id",
                    visit_id,
                )

    for hospital in HOSPITALS:
        for category, config in CATEGORY_REGISTRY.items():
            if not config["visit_linked"]:
                continue
            for index, record in enumerate(records_by_hospital[hospital][category], start=1):
                path = category_file(dataset_dir, hospital, category)
                source = f"{path} record {index}"
                visit_id = record.get("visit_id")
                visit = visits_by_id.get(visit_id)
                if visit is None:
                    missing_references.append({
                        "source": source,
                        "record_id": record.get(config["id_field"]),
                        "reference": "visit_id",
                        "value": visit_id,
                    })
                    continue
                if (
                    visit.get("patient_id") != record.get("patient_id")
                    or visit.get("hospital") != record.get("hospital")
                ):
                    invalid(
                        source,
                        "visit patient or hospital does not match record",
                        record.get(config["id_field"]),
                    )
                try:
                    record_time = _parse_timestamp(record[config["date_field"]])
                    visit_time = _parse_timestamp(visit["visit_datetime"])
                    if len(record[config["date_field"]].strip()) == 10:
                        precedes_visit = record_time.date() < visit_time.date()
                    else:
                        precedes_visit = record_time < visit_time
                    if precedes_visit:
                        invalid(
                            source,
                            "record timestamp precedes its visit",
                            record.get(config["id_field"]),
                        )
                except (KeyError, TypeError, ValueError) as error:
                    invalid(
                        source,
                        f"unable to validate visit-linked timestamp: {error}",
                        record.get(config["id_field"]),
                    )

    primary_distribution = _validate_primary_distribution(
        patient_by_id, records_by_hospital, dataset_dir, invalid
    )
    visit_count = sum(
        len(records_by_hospital[hospital]["visits"]) for hospital in HOSPITALS
    )
    unique_visit_count = len(visits_by_id)
    if expected_visit_count is not None and visit_count != expected_visit_count:
        invalid(
            dataset_dir,
            f"expected {expected_visit_count} visits, found {visit_count}",
        )

    category_counts = {
        category: sum(
            len(records_by_hospital[hospital][category]) for hospital in HOSPITALS
        )
        for category in CATEGORY_REGISTRY
    }
    hospital_category_counts = {
        hospital: {
            category: len(records_by_hospital[hospital][category])
            for category in CATEGORY_REGISTRY
        }
        for hospital in HOSPITALS
    }
    all_records = [
        record
        for hospital in HOSPITALS
        for category in CATEGORY_REGISTRY
        for record in records_by_hospital[hospital][category]
    ]
    report = {
        "valid": not (invalid_records or duplicate_ids or missing_references),
        "patient_count": len(patients),
        "unique_patient_count": len(patient_by_id),
        "visit_count": visit_count,
        "unique_visit_count": unique_visit_count,
        "hospital_visit_counts": {
            hospital: hospital_category_counts[hospital]["visits"]
            for hospital in HOSPITALS
        },
        "category_record_counts": category_counts,
        "hospital_category_counts": hospital_category_counts,
        "missing_references": missing_references,
        "missing_reference_count": len(missing_references),
        "duplicate_ids": duplicate_ids,
        "duplicate_id_count": len(duplicate_ids),
        "invalid_records": invalid_records,
        "invalid_record_count": len(invalid_records),
        "synthetic_marker_validation": {
            "all_patients_synthetic": bool(patients)
            and all(
                isinstance(patient, dict) and patient.get("synthetic") is True
                for patient in patients
            ),
            "all_records_synthetic": bool(all_records)
            and all(_is_synthetic(record.get("synthetic")) for record in all_records),
            "record_count_checked": len(all_records),
        },
        "primary_hospital_distribution": primary_distribution,
    }
    return report


def validate_dataset(
    dataset_dir,
    expected_patient_count=EXPECTED_PATIENT_COUNT,
    expected_visit_count=EXPECTED_VISIT_COUNT,
):
    report = inspect_dataset(
        dataset_dir,
        expected_patient_count=expected_patient_count,
        expected_visit_count=expected_visit_count,
    )
    if not report["valid"]:
        raise DatasetValidationError(report)
    return report


def _validate_category_values(category, record, source, record_id, invalid):
    if category == "visits":
        if record.get("status") not in VALID_VISIT_STATUSES:
            invalid(source, "invalid visit status", record_id)
        try:
            if int(record["length_of_stay_days"]) < 0:
                raise ValueError("length_of_stay_days cannot be negative")
        except (KeyError, TypeError, ValueError) as error:
            invalid(source, error, record_id)
    elif category == "medications":
        if record.get("medication_status") not in VALID_MEDICATION_STATUSES:
            invalid(source, "invalid medication_status", record_id)
        if record.get("end_date"):
            try:
                if _parse_timestamp(record["end_date"]) < _parse_timestamp(record["start_date"]):
                    invalid(source, "end_date precedes start_date", record_id)
            except (TypeError, ValueError) as error:
                invalid(source, f"invalid medication date: {error}", record_id)
    elif category == "lab_results":
        status = record.get("result_status")
        if status not in VALID_RESULT_STATUSES:
            invalid(source, "invalid result_status", record_id)
        if status == "pending" and record.get("result_value"):
            invalid(source, "pending result must not contain a result value", record_id)
        if status == "final" and not record.get("result_value"):
            invalid(source, "final result must contain a result value", record_id)
    elif category == "procedures":
        if record.get("procedure_status") not in VALID_PROCEDURE_STATUSES:
            invalid(source, "invalid procedure_status", record_id)
    elif category == "allergies":
        if record.get("allergy_status") not in VALID_ALLERGY_STATUSES:
            invalid(source, "invalid allergy_status", record_id)
    elif category == "vital_signs":
        for field, (minimum, maximum) in VITAL_RANGES.items():
            value = record.get(field)
            if value == "":
                continue
            try:
                numeric_value = float(value)
                if not minimum <= numeric_value <= maximum:
                    raise ValueError(f"{field} is outside the permitted range")
            except (TypeError, ValueError) as error:
                invalid(source, error, record_id)


def _validate_primary_distribution(patient_by_id, records_by_hospital, dataset_dir, invalid):
    visits_by_patient = {patient_id: [] for patient_id in patient_by_id}
    for hospital in HOSPITALS:
        for visit in records_by_hospital[hospital]["visits"]:
            patient_id = visit.get("patient_id")
            if isinstance(patient_id, str) and patient_id in visits_by_patient:
                visits_by_patient[patient_id].append(visit)

    per_patient = {}
    primary_visits = 0
    total_visits = 0
    patients_with_visits = 0
    matching_patients = 0
    for patient_id, patient in patient_by_id.items():
        visits = visits_by_patient[patient_id]
        if not visits:
            continue
        patients_with_visits += 1
        total = len(visits)
        actual = sum(
            visit.get("hospital") == patient.get("primary_hospital")
            for visit in visits
        )
        expected = math.floor(total * PRIMARY_HOSPITAL_SHARE + 0.5)
        exact_expected = total * PRIMARY_HOSPITAL_SHARE
        acceptable_counts = {expected}
        if math.isclose(
            exact_expected - math.floor(exact_expected),
            0.5,
        ):
            acceptable_counts.add(math.floor(exact_expected))
        primary_visits += actual
        matches = actual in acceptable_counts
        matching_patients += matches
        per_patient[patient_id] = {
            "visit_count": total,
            "primary_hospital": patient["primary_hospital"],
            "primary_visit_count": actual,
            "expected_primary_visit_count": expected,
            "acceptable_rounded_primary_counts": sorted(acceptable_counts),
            "primary_share": actual / total,
            "matches_rounded_85_15_policy": matches,
        }
        if not matches:
            invalid(
                dataset_dir,
                f"patient {patient_id} has {actual}/{total} primary-hospital visits; "
                f"expected {expected}",
                patient_id,
            )
    total_visits = sum(
        len(records_by_hospital[hospital]["visits"]) for hospital in HOSPITALS
    )
    assigned_visits = sum(len(visits) for visits in visits_by_patient.values())
    return {
        "target_primary_share": PRIMARY_HOSPITAL_SHARE,
        "total_visits": total_visits,
        "visits_with_known_patients": assigned_visits,
        "primary_hospital_visits": primary_visits,
        "overall_primary_hospital_share": (
            primary_visits / assigned_visits if assigned_visits else 0
        ),
        "patients_with_visits": patients_with_visits,
        "patients_meeting_rounded_policy": matching_patients,
        "per_patient": per_patient,
    }


def _is_synthetic(value):
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _parse_timestamp(value):
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp is empty")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed
