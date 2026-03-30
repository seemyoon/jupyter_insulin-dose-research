from sqlalchemy import exists
from sqlalchemy.orm import sessionmaker

from db.engine import engine
from db.models import Patient, Insulin, DiabetesTablets, \
    TakingInsulin, Hospitalization, Measurement, DietaryIntake, TakingDiabetesTablet, PatientAdditionalDrug, \
    PatientComorbidities, AdditionalDrug, Comorbidities


class Repository:
    def __init__(self):
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def get_patients_td(self):  # td - train dataset
        res = (
            self.session
            .query(Patient)
            # .filter(DatasetPartition.name == 'train')
            .filter(
                ~exists().where(Hospitalization.patient_id == Patient.id)
            )
            .all()
        )

        return res

    def get_patient_drugs_map(self):
        query = (
            self.session
            .query(PatientAdditionalDrug.patient_id, PatientAdditionalDrug.drug_id)
            .select_from(Patient)
            .join(PatientAdditionalDrug, Patient.id == PatientAdditionalDrug.patient_id)
            # .filter(DatasetPartition.name == 'train')
            .filter(
                ~exists().where(Hospitalization.patient_id == Patient.id)
            )
            .all()
        )

        mapping = {}

        for pid, drug_id in query:
            mapping.setdefault(pid, []).append(drug_id)

        return mapping

    def get_patient_comorbities_map(self):
        query = (
            self.session
            .query(PatientComorbidities.patient_id, PatientComorbidities.comorbidity_id)
            .select_from(Patient)
            .join(PatientComorbidities, Patient.id == PatientComorbidities.patient_id)
            #             .filter(DatasetPartition.name == 'train')
            .filter(
                ~exists().where(Hospitalization.patient_id == Patient.id)
            )
            .all()
        )

        mapping = {}

        for pid, comorb_id in query:
            mapping.setdefault(pid, []).append(comorb_id)

        return mapping

    def get_measurements(self):
        return (
            self.session.query(Measurement)
            .join(Measurement.patient)
            .filter(
                ~exists().where(
                    Hospitalization.patient_id == Measurement.patient_id
                )
            )
            .all()
        )

    def get_dietary(self):
        return (
            self.session
            .query(DietaryIntake)
            .join(DietaryIntake.patient)
            #             .filter(DatasetPartition.name == 'train')
            .filter(
                ~exists().where(
                    Hospitalization.patient_id == DietaryIntake.patient_id
                )
            )
            .all()
        )

    def get_taking_tablets(self):
        return (
            self.session.query(TakingDiabetesTablet)
            .join(TakingDiabetesTablet.patient)
            .filter(
                ~exists().where(
                    Hospitalization.patient_id == TakingDiabetesTablet.patient_id
                )
            )
            .all()
        )

    def get_taking_insulin(self):
        return (
            self.session.query(TakingInsulin)
            .join(TakingInsulin.patient)
            .filter(
                ~exists().where(
                    Hospitalization.patient_id == TakingInsulin.patient_id
                )
            )
            .all()
        )

    def get_insulin_list(self):
        query = (self.session
                 .query(Insulin.id, Insulin.name)
                 .all())

        mapping = {}
        for pid, value in query:
            if pid in mapping.keys():
                continue
            else:
                mapping[pid] = value

        return mapping

    def get_tablets_list(self):
        query = (self.session
                 .query(DiabetesTablets.id, DiabetesTablets.name)
                 .all())

        mapping = {}
        for pid, value in query:
            if pid in mapping.keys():
                continue
            else:
                mapping[pid] = value

        return mapping


    def get_additional_drugs_list(self):
        query = self.session.query(AdditionalDrug.id, AdditionalDrug.name).all()
        return {row[0]: row[1] for row in query}

    def get_comorbidities_list(self):
        query = self.session.query(Comorbidities.id, Comorbidities.name).all()
        return {row[0]: row[1] for row in query}


if __name__ == '__main__':
    rep = Repository()