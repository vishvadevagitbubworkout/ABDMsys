from services.dataset_registry import CATEGORY_REGISTRY, HOSPITALS
from services.patient_record_service import (
    PatientRecordRetrievalError,
    retrieve_patient_records,
)
from services.result_formatter import format_patient_result


def retrieve_patient_records_for_agent(patient_id, dataset_dir=None):
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise PatientRecordRetrievalError("A patient ID is required")

    retrieval = (
        retrieve_patient_records(patient_id)
        if dataset_dir is None
        else retrieve_patient_records(patient_id, dataset_dir)
    )
    for hospital in HOSPITALS:
        for category in CATEGORY_REGISTRY:
            for record in retrieval["hospitals"][hospital][category]:
                if record.get("patient_id") != patient_id:
                    raise PatientRecordRetrievalError(
                        "Retrieval returned a record for a different patient"
                    )
                if record.get("hospital") != hospital:
                    raise PatientRecordRetrievalError(
                        "Retrieval returned a record with an incorrect hospital source"
                    )

    return format_patient_result(retrieval)
