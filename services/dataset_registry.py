from pathlib import Path


HOSPITALS = ("Hospital_A", "Hospital_B", "Hospital_C")
PATIENTS_FILENAME = "patients.json"
EXPECTED_PATIENT_COUNT = 500
EXPECTED_VISIT_COUNT = 2000
PRIMARY_HOSPITAL_SHARE = 0.85
SCHEMA_VERSION = "2.0"

CATEGORY_REGISTRY = {
    "visits": {
        "directory": "visit_records",
        "filename": "visits.csv",
        "format": "csv",
        "id_field": "visit_id",
        "date_field": "visit_datetime",
        "visit_linked": False,
        "fields": (
            "visit_id",
            "patient_id",
            "hospital",
            "visit_datetime",
            "department",
            "visit_type",
            "reason_for_visit",
            "status",
            "length_of_stay_days",
            "synthetic",
        ),
    },
    "diagnoses": {
        "directory": "diagnosis_records",
        "filename": "diagnoses.csv",
        "format": "csv",
        "id_field": "diagnosis_id",
        "date_field": "diagnosis_datetime",
        "visit_linked": True,
        "fields": (
            "diagnosis_id",
            "patient_id",
            "visit_id",
            "hospital",
            "diagnosis_code",
            "diagnosis_name",
            "diagnosis_type",
            "diagnosis_datetime",
            "clinical_status",
            "synthetic",
        ),
    },
    "medications": {
        "directory": "medication_records",
        "filename": "medications.csv",
        "format": "csv",
        "id_field": "medication_id",
        "date_field": "start_date",
        "visit_linked": True,
        "fields": (
            "medication_id",
            "patient_id",
            "visit_id",
            "hospital",
            "medication_name",
            "dosage",
            "frequency",
            "route",
            "start_date",
            "end_date",
            "medication_status",
            "prescribed_by",
            "synthetic",
        ),
    },
    "lab_results": {
        "directory": "laboratory_records",
        "filename": "lab_results.csv",
        "format": "csv",
        "id_field": "lab_id",
        "date_field": "test_datetime",
        "visit_linked": True,
        "fields": (
            "lab_id",
            "patient_id",
            "visit_id",
            "hospital",
            "test_name",
            "test_code",
            "result_value",
            "unit",
            "reference_range",
            "result_status",
            "test_datetime",
            "synthetic",
        ),
    },
    "procedures": {
        "directory": "procedure_records",
        "filename": "procedures.csv",
        "format": "csv",
        "id_field": "procedure_id",
        "date_field": "procedure_datetime",
        "visit_linked": True,
        "fields": (
            "procedure_id",
            "patient_id",
            "visit_id",
            "hospital",
            "procedure_name",
            "procedure_code",
            "procedure_datetime",
            "procedure_status",
            "notes",
            "synthetic",
        ),
    },
    "clinical_notes": {
        "directory": "clinical_notes",
        "filename": "clinical_notes.json",
        "format": "json",
        "id_field": "note_id",
        "date_field": "note_datetime",
        "visit_linked": True,
        "fields": (
            "note_id",
            "patient_id",
            "visit_id",
            "hospital",
            "note_datetime",
            "note_type",
            "author_role",
            "chief_complaint",
            "clinical_observations",
            "assessment",
            "plan",
            "discharge_summary",
            "synthetic",
        ),
    },
    "allergies": {
        "directory": "allergy_records",
        "filename": "allergies.csv",
        "format": "csv",
        "id_field": "allergy_id",
        "date_field": "recorded_datetime",
        "visit_linked": False,
        "fields": (
            "allergy_id",
            "patient_id",
            "hospital",
            "allergen",
            "reaction",
            "severity",
            "recorded_datetime",
            "allergy_status",
            "synthetic",
        ),
    },
    "vital_signs": {
        "directory": "vital_records",
        "filename": "vital_signs.csv",
        "format": "csv",
        "id_field": "vital_id",
        "date_field": "measurement_datetime",
        "visit_linked": True,
        "fields": (
            "vital_id",
            "patient_id",
            "visit_id",
            "hospital",
            "measurement_datetime",
            "systolic_bp",
            "diastolic_bp",
            "heart_rate",
            "temperature_c",
            "respiratory_rate",
            "spo2",
            "weight_kg",
            "height_cm",
            "synthetic",
        ),
    },
}

VISIT_LINKED_CATEGORIES = tuple(
    category
    for category, config in CATEGORY_REGISTRY.items()
    if config["visit_linked"]
)
PATIENTS_FILE = Path("dataset") / PATIENTS_FILENAME


def category_file(dataset_dir, hospital, category):
    config = CATEGORY_REGISTRY[category]
    return Path(dataset_dir) / hospital / config["directory"] / config["filename"]
