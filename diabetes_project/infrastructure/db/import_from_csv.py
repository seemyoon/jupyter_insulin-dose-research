import pandas as pd

from sqlalchemy.orm import sessionmaker

from diabetes_project.infrastructure.db.engine import engine
from diabetes_project.infrastructure.db.models import Patient, PatientMedicalStatic, AdditionalDrugs

class DataImporter:
    def __init__(self):
        Session = sessionmaker(bind=engine)  # creates a session factory class that knows what engine it is connecting to.
        self.session = Session()  # start a new session (connection to the database) with which we work.

    def import_from_data(self, file_path):
        pass
        # try:
        #     df = pd.read_csv(file_path)

            # for patient_id, group in df.groupby('Patient Number')






