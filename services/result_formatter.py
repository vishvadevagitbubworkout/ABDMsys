from datetime import datetime, timezone

from services.dataset_registry import CATEGORY_REGISTRY, HOSPITALS


def format_patient_result(retrieval):
    hospitals = retrieval["hospitals"]
    records_by_category = {category: [] for category in CATEGORY_REGISTRY}
    hospital_summary = {}

    for hospital in HOSPITALS:
        hospital_summary[hospital] = {}
        for category in CATEGORY_REGISTRY:
            records = hospitals[hospital][category]
            records.sort(key=lambda record: _record_datetime(record, category))
            hospital_summary[hospital][category] = len(records)
            records_by_category[category].extend(records)

    for category, records in records_by_category.items():
        records.sort(key=lambda record: _record_datetime(record, category))

    record_summary = {
        category: len(records)
        for category, records in records_by_category.items()
    }
    hospital_count = sum(
        any(count for count in hospital_summary[hospital].values())
        for hospital in HOSPITALS
    )
    return {
        "patient_id": retrieval["patient_id"],
        "synthetic": bool(retrieval["patient"].get("synthetic")),
        "hospital_count": hospital_count,
        "hospitals_searched": len(HOSPITALS),
        "record_summary": record_summary,
        "hospital_summary": hospital_summary,
        "hospitals": hospitals,
        "records": records_by_category,
    }


def _record_datetime(record, category):
    field = CATEGORY_REGISTRY[category]["date_field"]
    value = record[field]
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
