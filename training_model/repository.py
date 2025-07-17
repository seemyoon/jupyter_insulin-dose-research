from sqlalchemy.orm import sessionmaker

from db.engine import engine
from db.models import Patient, DatasetPartition, AdditionalDrugs, Comorbidities


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
            .all()
        )

    def get_patient_drugs_map(self):
        query = (
            self.session
            .query(Patient.id, AdditionalDrugs.id)
            .join(AdditionalDrugs)
        )

        mapping = {}

        for pid, value in query:
            mapping.setdefault(pid, []).append(value)

        return mapping

    def get_patient_comorbities_map(self):
        query = (self.session
                 .query(Patient.id, Comorbidities.id)
                 .join(Comorbidities)
                 )

        mapping = {}

        for pid, value in query:
            mapping.setdefault(pid, []).append(value)

        return mapping
