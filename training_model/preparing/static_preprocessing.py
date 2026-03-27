import torch
from sklearn.preprocessing import MinMaxScaler


class StaticProcessing:

    @staticmethod
    def get_unique_entities(entity_list):
        """
        accepts a list of lists of IDs (e.g. drugs or diseases),
        returns a dictionary of {ID: index} for building embeddings.

        now if a patient has a drug with ID 12, we know that for the model this is index 2.
        the model works with such indices (e.g. embedding(2)), not with the original IDs, which can be large and messy.
        """
        unique_ids = set(entity for entity_sublist in entity_list for entity in entity_sublist)
        return {entity_id: idx + 1 for idx, entity_id in enumerate(sorted(unique_ids))}

    @staticmethod
    def fit_scaler(features):
        """Fit a MinMaxScaler on features (should be training set only) and return it."""
        scaler = MinMaxScaler()
        scaler.fit(features)
        return scaler

    @staticmethod
    def transform_features(features, scaler):
        """Transform features using a pre-fitted scaler."""
        return scaler.transform(features)

    @staticmethod
    def extract_patient_data(patients, unique_drugs, unique_comorbities,
                             patient_to_drugs, patient_to_comorbities):
        """
        Extract raw (unnormalized) static features and embedding indices per patient.

        :return: (patient_ids, raw_features, drug_indices, comorb_indices)
        """
        patient_ids = []
        raw_static_features = []
        drug_indices = []
        comorb_indices = []

        for patient in patients:
            patient_ids.append(patient.id)

            features = [
                patient.gender,
                patient.age,
                patient.height,
                patient.weight,
                patient.smoking_history or 0,
                patient.alcohol_drinking_history or 0
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
            drug_indices.append(drug_idx or [0])

            comorb_ids = patient_to_comorbities.get(patient.id, [])
            comorb_idx = [unique_comorbities[comorb_id] for comorb_id in comorb_ids if comorb_id in unique_comorbities]
            comorb_indices.append(comorb_idx or [0])

        return patient_ids, raw_static_features, drug_indices, comorb_indices

    @staticmethod
    def build_static_dict(patient_ids, raw_features, drug_indices, comorb_indices, scaler):
        """
        Build per-patient dict with normalized features and padded index tensors.

        :param scaler: a fitted MinMaxScaler (fit on training patients only)
        :return: {patient_id: {'static': Tensor, 'drug_idx': LongTensor, 'comorb_idx': LongTensor}}
        """
        normalized = scaler.transform(raw_features)

        max_drug_len = max(len(lst) for lst in drug_indices)
        max_comorb_len = max(len(lst) for lst in comorb_indices)

        static_dict = {}
        for i, pid in enumerate(patient_ids):
            static_tensor = torch.tensor(normalized[i], dtype=torch.float32)

            drug_padded = torch.zeros(max_drug_len, dtype=torch.long)
            drug_padded[:len(drug_indices[i])] = torch.tensor(drug_indices[i], dtype=torch.long)

            comorb_padded = torch.zeros(max_comorb_len, dtype=torch.long)
            comorb_padded[:len(comorb_indices[i])] = torch.tensor(comorb_indices[i], dtype=torch.long)

            static_dict[pid] = {
                'static': static_tensor,
                'drug_idx': drug_padded,
                'comorb_idx': comorb_padded,
            }

        return static_dict
