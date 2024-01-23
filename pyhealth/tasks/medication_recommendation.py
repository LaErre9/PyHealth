from pyhealth.data import Patient, Visit


# prova prova
def medication_recommendation_mimic3_fn(patient: Patient):
    """Processes a single patient for the medication recommendation task.

    Medication recommendation aims at recommending a set of medications given the patient health
    history  (e.g., diagnosis and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key, like this
            {
                "patient_id": xxx,
                "visit_id": xxx,
                "diagnosis": [list of diag in visit 1, list of diag in visit 2, ..., list of diag in visit N],
                "procedures": [list of prod in visit 1, list of prod in visit 2, ..., list of prod in visit N],
                "drugs_hist": [list of medication in visit 1, list of medication in visit 2, ..., list of medication in visit (N-1)],
                "medications": list of medication in visit N, # this is the predicted target
            }

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> mimic3_base = MIMIC3Dataset(
        ...    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...    code_mapping={"ICD9CM": "CCSCM"},
        ... )
        >>> from pyhealth.tasks import medication_recommendation_mimic3_fn
        >>> mimic3_sample = mimic3_base.set_task(medication_recommendation_mimic3_fn)
        >>> mimic3_sample.samples[0]
        {
            'visit_id': '174162',
            'patient_id': '107',
            'diagnosis': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'medications': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        diagnosis = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        medications = visit.get_code_list(table="PRESCRIPTIONS")
        symptoms = visit.get_code_list(table='NOTEEVENTS_ICD')

        # ATC 4 level
        medications = [medication[:5] for medication in medications]
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
                "medications": medications,
                "drugs_hist": medications,
                "symptoms": symptoms,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []

    # add history
    samples[0]["diagnosis"] = [samples[0]["diagnosis"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["symptoms"] = [samples[0]["symptoms"]]
    

    for i in range(1, len(samples)):
        samples[i]["diagnosis"] = samples[i - 1]["diagnosis"] + [
            samples[i]["diagnosis"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["symptoms"] = samples[i - 1]["symptoms"] + [
            samples[i]["symptoms"]
        ]
        

    # remove the target medication from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = []

    return samples



def medication_recommendation_mimic4_fn(patient: Patient):
    """Processes a single patient for the medication recommendation task.

    Medication recommendation aims at recommending a set of medications given the patient health
    history  (e.g., diagnosis and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key
            {
                "patient_id": xxx,
                "visit_id": xxx,
                "diagnosis": [list of diag in visit 1, list of diag in visit 2, ..., list of diag in visit N],
                "procedures": [list of prod in visit 1, list of prod in visit 2, ..., list of prod in visit N],
                "drugs_hist": [list of medication in visit 1, list of medication in visit 2, ..., list of medication in visit (N-1)],
                "medications": list of medication in visit N, # this is the predicted target
            }

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> mimic4_base = MIMIC4Dataset(
        ...     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...     tables=["diagnoses_icd", "procedures_icd"],
        ...     code_mapping={"ICD10PROC": "CCSPROC"},
        ... )
        >>> from pyhealth.tasks import medication_recommendation_mimic4_fn
        >>> mimic4_sample = mimic4_base.set_task(medication_recommendation_mimic4_fn)
        >>> mimic4_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'diagnosis': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': [['2', '3', '4']]}]
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        diagnosis = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        medications = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        medications = [medication[:4] for medication in medications]
        # exclude: visits without condition, procedure, or medication code
        if len(diagnosis) * len(procedures) * len(medications) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "diagnosis": diagnosis,
                "procedures": procedures,
                "medications": medications,
                "drugs_hist": medications,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["diagnosis"] = [samples[0]["diagnosis"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]

    for i in range(1, len(samples)):
        samples[i]["diagnosis"] = samples[i - 1]["diagnosis"] + [
            samples[i]["diagnosis"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]

    # remove the target medication from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = []

    return samples


def medication_recommendation_eicu_fn(patient: Patient):
    """Processes a single patient for the medication recommendation task.

    Medication recommendation aims at recommending a set of medications given the patient health
    history  (e.g., diagnosis and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> eicu_base = eICUDataset(
        ...     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication"],
        ...     code_mapping={},
        ...     dev=True
        ... )
        >>> from pyhealth.tasks import medication_recommendation_eicu_fn
        >>> eicu_sample = eicu_base.set_task(medication_recommendation_eicu_fn)
        >>> eicu_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'diagnosis': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': [['2', '3', '4']]}]
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        diagnosis = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        medications = visit.get_code_list(table="medication")
        # exclude: visits without condition, procedure, or medication code
        if len(diagnosis) * len(procedures) * len(medications) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "diagnosis": diagnosis,
                "procedures": procedures,
                "medications": medications,
                "drugs_all": medications,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["diagnosis"] = [samples[0]["diagnosis"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["diagnosis"] = samples[i - 1]["diagnosis"] + [
            samples[i]["diagnosis"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples


def medication_recommendation_omop_fn(patient: Patient):
    """Processes a single patient for the medication recommendation task.

    Medication recommendation aims at recommending a set of medications given the patient health
    history  (e.g., diagnosis and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> omop_base = OMOPDataset(
        ...     root="https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2",
        ...     tables=["condition_occurrence", "procedure_occurrence"],
        ...     code_mapping={},
        ... )
        >>> from pyhealth.tasks import medication_recommendation_omop_fn
        >>> omop_sample = omop_base.set_task(medication_recommendation_eicu_fn)
        >>> omop_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'diagnosis': [['42', '109', '98', '663', '58', '51'], ['98', '663', '58', '51']], 'procedures': [['1'], ['2', '3']], 'label': [['2', '3', '4'], ['0', '1', '4', '5']]}]
    """

    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        diagnosis = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        medications = visit.get_code_list(table="drug_exposure")
        # exclude: visits without condition, procedure, or medication code
        if len(diagnosis) * len(procedures) * len(medications) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "diagnosis": diagnosis,
                "procedures": procedures,
                "medications": medications,
                "drugs_all": medications,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["diagnosis"] = [samples[0]["diagnosis"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["diagnosis"] = samples[i - 1]["diagnosis"] + [
            samples[i]["diagnosis"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples


if __name__ == "__main__":
    # from pyhealth.datasets import MIMIC3Dataset
    # base_dataset = MIMIC3Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    #     tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    #     dev=True,
    #     code_mapping={"ICD9CM": "CCSCM"},
    #     refresh_cache=False,
    # )
    # sample_dataset = base_dataset.set_task(task_fn=medication_recommendation_mimic3_fn)
    # sample_dataset.stat()
    # print(sample_dataset.available_keys)
    # print(sample_dataset.samples[0])

    from pyhealth.datasets import MIMIC4Dataset

    base_dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=medication_recommendation_mimic4_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)
    print(sample_dataset.samples[0])

    # from pyhealth.datasets import eICUDataset

    # base_dataset = eICUDataset(
    #     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
    #     tables=["diagnosis", "medication", "physicalExam"],
    #     dev=True,
    #     refresh_cache=False,
    # )
    # sample_dataset = base_dataset.set_task(task_fn=medication_recommendation_eicu_fn)
    # sample_dataset.stat()
    # print(sample_dataset.available_keys)

    # from pyhealth.datasets import OMOPDataset

    # base_dataset = OMOPDataset(
    #     root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
    #     tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
    #     dev=True,
    #     refresh_cache=False,
    # )
    # sample_dataset = base_dataset.set_task(task_fn=medication_recommendation_omop_fn)
    # sample_dataset.stat()
    # print(sample_dataset.available_keys)
