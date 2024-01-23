from pyhealth.data import Patient, Visit


def diagnosis_prediction_mimic3_fn(patient: Patient):
    """Processes a single patient for the diagnosis prediction task.

    Diagnosis prediction aims at predict a set of diagnosis given the patient health
    history  (e.g., symptoms, procedures and medications).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key, like this
            {
                "patient_id": xxx,
                "visit_id": xxx,
                "diagnosis": [list of diag in visit N], # this is the predicted target 
                "procedures": [list of prod in visit 1, list of prod in visit 2, ..., list of prod in visit N],
                "symptoms": [list of symptom in visit 1, list of symptom in visit 2, ..., list of symptom in visit N],
                "medications": [list of medication in visit 1, list of medication in visit 2, ..., list of medication in visit N]
            }
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        diagnosis = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        medications = visit.get_code_list(table="PRESCRIPTIONS")
        symptoms = visit.get_code_list(table='NOTEEVENTS_ICD')

        # ATC 3 level
        medications = [medication[:4] for medication in medications]
        # Diagnosis 3 level
        diagnosis = [condition[:3] for condition in diagnosis]
        # exclude: visits without condition, procedure, or medication code
        if len(diagnosis) * len(procedures) * len(medications) * len(symptoms) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "diagnosis": diagnosis,
                "procedures": procedures,
                "symptoms": symptoms,
                "medications": medications,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []

    # add history
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["symptoms"] = [samples[0]["symptoms"]]
    samples[0]["medications"] = [samples[0]["medications"]]
    

    for i in range(1, len(samples)):
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["symptoms"] = samples[i - 1]["symptoms"] + [
            samples[i]["symptoms"]
        ]
        samples[i]["medications"] = samples[i - 1]["medications"] + [
            samples[i]["medications"]
        ]


    return samples