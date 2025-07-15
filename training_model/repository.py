from sqlalchemy.orm import sessionmaker, Session

from db.engine import engine
from db.models import Patient, DatasetPartition
import torch


class Repository:
    def __init__(self):
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def get_patients_td(self):  # td - train dataset
        patients_noformat = (
            self.session
            .query(Patient)
            .join(DatasetPartition)
            .filter(DatasetPartition.name == 'train')
            .all()
        )

        return torch.tensor([[
            p.gender,
            p.age,
            p.height,
            p.weight,
            p.smoking_history,
            p.alcohol_drinking_history,
        ] for p in patients_noformat], dtype=torch.float32)

    def get_static_features_td(self, session: Session):
        pass

    def _get_patient_ids_td(self):
        return [
            pid for (pid,) in self
            .session
            .query(Patient.id)
            .join(DatasetPartition)
            .filter(DatasetPartition.name == 'train')
            .all()
        ]


if __name__ == '__main__':
    repo = Repository()
    repo.get_patients_td()
