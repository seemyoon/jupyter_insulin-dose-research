from locale import normalize

import torch
from sklearn.preprocessing import MinMaxScaler


class StaticProcessing:

    @staticmethod
    def get_unique_entities(entity_list):
        """
        accepts a list of lists of IDs (e.g. drugs or diseases),
        returns a dictionary of {ID: index} for building embeddings.

        now if a patient has a drug with ID 12, we know that for the model this is index 2.
        the model works with such indices (eg embedding(2)), not with the original IDs, which can be large and messy.
        """
        unique_ids = set(entity for entity_sublist in entity_list for entity in entity_sublist)  # {5, 12, 7}
        # return {entity_id: idx for idx, entity_id in enumerate(sorted(unique_ids))}  # {5: 0, 7: 1, 12: 2}
        return {entity_id: idx+1 for idx, entity_id in enumerate(sorted(unique_ids))}  # {5: 0, 7: 1, 12: 2}

    @staticmethod
    def normalize_features(features):
        scaler = MinMaxScaler()
        return scaler.fit_transform(features)

    @staticmethod
    def get_static_tensor_with_embeddings(patients, unique_drugs, unique_comorbities, patient_to_drugs,
                                          patient_to_comorbities):
        """
        :param patients:
        :param unique_drugs:
        :param unique_comorbities:
        :param patient_to_drugs:
        :param patient_to_comorbities:
        :return:
        """

        raw_static_features = []
        drug_indices = []
        comorb_indices = []

        for patient in patients:
            features = [
                patient.gender,
                patient.age,
                patient.height,
                patient.weight,
                patient.smoking_history,
                patient.alcohol_drinking_history
            ]

            med = patient.medical_static

            features += [
                med.diabetes_type,
                med.diabetes_duration_years,
                med.fasting_glucose,
                med.postprandial_glucose,
                med.fasting_c_peptide,
                med.postprandial_c_peptide,
                med.fasting_insulin,
                med.postprandial_insulin,
                med.hba1c,
                med.glycated_albumin,
                med.total_cholesterol,
                med.triglyceride,
                med.hdl,
                med.ldl,
                med.creatinine,
                med.egfr,
                med.uric_acid,
                med.bun,
            ]

            raw_static_features.append(features)

            drug_ids = patient_to_drugs.get(patient.id, [])
            drug_idx = [unique_drugs[drug_id] for drug_id in drug_ids if drug_id in unique_drugs]
            drug_indices.append(drug_idx)

            comorb_ids = patient_to_comorbities.get(patient.id, [])
            comorb_idx = [unique_comorbities[comorb_id] for comorb_id in comorb_ids if comorb_id in unique_comorbities]
            comorb_indices.append(comorb_idx)

        normalize_raw_static_features = StaticProcessing.normalize_features(raw_static_features)
        static_tensor = torch.tensor(normalize_raw_static_features, dtype=torch.float32)

        return static_tensor, drug_indices, comorb_indices
