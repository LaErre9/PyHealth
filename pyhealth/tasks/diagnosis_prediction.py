from pyhealth.data import Patient, Visit


def diagnosis_prediction_mimic3_fn(patient: Patient):
    """Processes a single patient for the diagnosis prediction task.

    Diagnosis prediction aims at predict a set of diagnosis given the patient health
    history  (e.g., symptoms, procedures and drugs).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key, like this
            {
                "patient_id": xxx,
                "visit_id": xxx,
                "conditions": [list of diag in visit N], # this is the predicted target 
                "procedures": [list of prod in visit 1, list of prod in visit 2, ..., list of prod in visit N],
                "symptoms": [list of symptom in visit 1, list of symptom in visit 2, ..., list of symptom in visit N],
                "drugs": [list of drug in visit 1, list of drug in visit 2, ..., list of drug in visit N]
            }
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        symptoms = visit.get_code_list(table='NOTEEVENTS_ICD')

        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) * len(symptoms) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "symptoms": symptoms,
                "drugs": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []

    # add history
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["symptoms"] = [samples[0]["symptoms"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    

    for i in range(1, len(samples)):
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["symptoms"] = samples[i - 1]["symptoms"] + [
            samples[i]["symptoms"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]


    return samples