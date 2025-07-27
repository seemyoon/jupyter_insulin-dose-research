from sqlalchemy.orm import sessionmaker

from db.engine import engine
from db.models import Patient, DatasetPartition, AdditionalDrugs, Comorbidities, Insulin, DiabetesTablets, \
    TakingInsulin, Hospitalization, Measurement, DietaryIntake, TakingDiabetesTablet
from sqlalchemy import exists


class Repository:
    def __init__(self):
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def get_patients_td(self):  # td - train dataset
        return (
            self.session
            .query(Patient)
            .join(DatasetPartition)
            .filter(DatasetPartition.name == 'train')
            .filter(
                ~exists().where(Hospitalization.patient_id == Patient.id)
            )
            .all()
        )

    def get_patient_drugs_map(self):
        query = (
            self.session
            .query(Patient.id, AdditionalDrugs.id)
            .join(AdditionalDrugs)
            .join(DatasetPartition)
            .filter(DatasetPartition.name == 'train')
            .filter(
                ~exists().where(Hospitalization.patient_id == Patient.id)
            )
        )

        mapping = {}

        for pid, value in query:
            mapping.setdefault(pid, []).append(value)

        return mapping

    def get_patient_comorbities_map(self):
        query = (
            self.session
            .query(Patient.id, Comorbidities.id)
            .join(Comorbidities)
            .join(DatasetPartition)
            .filter(DatasetPartition.name == 'train')
            .filter(
                ~exists().where(Hospitalization.patient_id == Patient.id)
            )

        )

        mapping = {}

        for pid, value in query:
            mapping.setdefault(pid, []).append(value)

        return mapping

    def get_taking_insulin(self):
        return (
            self.session.query(TakingInsulin)
            .join(TakingInsulin.patient)
            .join(Patient.dataset_partition)
            .filter(DatasetPartition.name == 'train')
            .filter(
                ~exists().where(
                    Hospitalization.patient_id == TakingInsulin.patient_id
                )
            )
            .all()
        )

    def get_measurements(self):
        return (
            self.session.query(Measurement)
            .join(Measurement.patient)
            .join(Patient.dataset_partition)
            .filter(DatasetPartition.name == 'train')
            .filter(
                ~exists().where(
                    Hospitalization.patient_id == TakingInsulin.patient_id
                )
            )
            .all()
        )

    def get_dietary(self):
        return (
            self.session
            .query(DietaryIntake)
            .join(DietaryIntake.patient)
            .filter(DatasetPartition.name == 'train')
            .filter(
                ~exists().where(
                    Hospitalization.patient_id == TakingInsulin.patient_id
                )
            )
            .all()
        )

    def get_taking_tablets(self):
        return (
            self.session.query(TakingDiabetesTablet)
            .join(TakingDiabetesTablet.patient)
            .join(Patient.dataset_partition)
            .filter(DatasetPartition.name == 'train')
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


if __name__ == '__main__':
    rep = Repository()
    insulin_list = rep.get_tablets_list()
    print(insulin_list)
