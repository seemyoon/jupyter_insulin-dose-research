import pandas as pd
from sqlalchemy.orm import sessionmaker

from db.engine import engine
from db.models import Patient, PatientMedicalStatic, AdditionalDrugs, Comorbidities, \
    DatasetPartition, DietaryIntake, Measurement, Insulin, DiabetesTablets, TakingInsulin, TakingDiabetesTablet
from utils.convert_python_format import to_python_format


class DataImporter:
    def __init__(self, partition_name: str):
        Session = sessionmaker(
            bind=engine)  # creates a session factory class that knows what engine it is connecting to.
        self.session = Session()  # start a new session (connection to the database) with which we work.
        self.partition = self._get_or_create_partition(partition_name)

    additional_drug_columns = [
        "has_ace_inhibitors", "has_angioprotectors", "has_antianginal", "has_antiarrhythmic",
        "has_antibiotics", "has_antihypertensives", "has_antithrombotic", "has_arb",
        "has_calcium_channel_blockers", "has_circulatory_support", "has_gastroprotective",
        "has_gout_treatment", "has_hepatoprotector", "has_hypolipidemic", "has_immunomodulators",
        "has_kidney_support", "has_laxatives", "has_minerals_and_vitamins", "has_neuroprotectors",
        "has_pancreatic", "has_probiotics", "has_psychotropic", "has_thyroid_diseases",
        "has_urological_drugs", "has_vasodilators", "has_vestibular_disorders",
        "has_autoimmune_diseases", "has_cardiovascular_diseases", "has_dental_diseases",
        "has_diseases_of_the_musculoskeletal_system", "has_diseases_of_the_stomach_and_intestines",
        "has_electrolyte_and_mineral_disorders"
    ]

    comorbidities_columns = ['has_endocrine_diseases', 'has_eye_diseases', 'has_gallbladder_diseases',
                             'has_gynecological_diseases', 'has_hematologic_disorders',
                             'has_infectious_diseases', 'has_kidney_diseases', 'has_liver_diseases',
                             'has_male_reproductive_diseases', 'has_neurological_and_psychiatric_diseases',
                             'has_oncology', 'has_diabetic_microvascular_complications',
                             'has_diabetic_macrovascular__complications', 'has_acute_diabetic_complications',
                             'has_hypoglycemia']

    insulin_columns = [
        'dose_insulin_glargine', 'dose_novolin_50r', 'dose_gansulin_r', 'dose_novolin_r', 'dose_humulin_r',
        'dose_insulin_glulisine', 'dose_insulin_detemir', 'dose_insulin_aspart_70_30', 'dose_novolin_30r',
        'dose_scilin_m30', 'dose_gansulin_40r', 'dose_humulin_70_30', 'dose_insulin_aspart',
        'dose_insulin_degludec'
    ]

    def import_from_data(self, file_path: str):
        try:
            df = pd.read_csv(file_path)

            for patient_id, group in df.groupby('Patient Number'):
                first_row = group.iloc[0]
                self._import_patient(patient_id, first_row)
                self._import_medical_static(patient_id, first_row)
                self._import_additional_drugs(patient_id, first_row)
                self._import_comorbidities(patient_id, first_row)

                self._import_dietary_intake(patient_id, group)
                self._import_measurement(patient_id, group)
                self._import_insulin_dose(patient_id, group)
                self._import_diabetes_tablet(patient_id, group)

            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise TypeError(f"Error during import from {file_path}: {e}")

    @staticmethod
    def _extract_time_fields(row):
        return {
            'year': to_python_format(row.get("year_treat")),
            'month': to_python_format(row.get("month_treat")),
            'day': to_python_format(row.get("day_treat")),
            'hour': to_python_format(row.get("hour_of_day_treat")),
            'minute': to_python_format(row.get("minute_treat")),
        }

    def _import_patient(self, patient_id, first_row):
        try:
            patient_exist = self.session.query(Patient).filter_by(id=str(patient_id)).first()
            if patient_exist: return

            patient = Patient(
                id=str(patient_id),
                gender=to_python_format(first_row.get('Gender (Female=1, Male=2)')),
                age=to_python_format(first_row.get('Age (years)')),
                height=to_python_format(first_row.get('Height (m)')),
                weight=to_python_format(first_row.get('Weight (kg)')),
                smoking_history=to_python_format(first_row.get('Smoking History (pack year)')),
                alcohol_drinking_history=to_python_format(
                    first_row.get('Alcohol Drinking History (drinker/non-drinker)')),
                dataset_partition_id=to_python_format(self.partition.id)
            )
            self.session.add(patient)
        except Exception as e:
            raise TypeError(f"[ERROR] import_patient for ID {patient_id}: {e} in method import_patient")

    def _import_medical_static(self, patient_id, first_row):
        try:
            medical_static = PatientMedicalStatic(
                patient_id=str(patient_id),
                diabetes_type=to_python_format(first_row.get("Type of Diabetes")),
                diabetes_duration_years=to_python_format(first_row.get("Duration of Diabetes (years)")),
                fasting_glucose=to_python_format(first_row.get("Fasting Plasma Glucose (mg/dl)")),
                postprandial_glucose=to_python_format(first_row.get("2-hour Postprandial Plasma Glucose (mg/dl)")),
                fasting_c_peptide=to_python_format(first_row.get("Fasting C-peptide (nmol/L)")),
                postprandial_c_peptide=to_python_format(first_row.get("2-hour Postprandial C-peptide (nmol/L)")),
                fasting_insulin=to_python_format(first_row.get("Fasting Insulin (pmol/L)")),
                postprandial_insulin=to_python_format(first_row.get("2-hour Postprandial Insulin (pmol/L)")),
                hba1c=to_python_format(first_row.get("HbA1c (mmol/mol)")),
                glycated_albumin=to_python_format(first_row.get("Glycated Albumin (%)")),
                total_cholesterol=to_python_format(first_row.get("Total Cholesterol (mmol/L)")),
                triglyceride=to_python_format(first_row.get("Triglyceride (mmol/L)")),
                hdl=to_python_format(first_row.get("High-Density Lipoprotein Cholesterol (mmol/L)")),
                ldl=to_python_format(first_row.get("Low-Density Lipoprotein Cholesterol (mmol/L)")),
                creatinine=to_python_format(first_row.get("Creatinine (umol/L)")),
                egfr=to_python_format(first_row.get("Estimated Glomerular Filtration Rate  (ml/min/1.73m2)")),
                uric_acid=to_python_format(first_row.get("Uric Acid (mmol/L)")),
                bun=to_python_format(first_row.get("Blood Urea Nitrogen (mmol/L)")),
            )
            self.session.add(medical_static)
        except Exception as e:
            raise TypeError(f"[ERROR] import_patient for ID {patient_id}: {e} in method import_medical_static")

    def _import_additional_drugs(self, patient_id, first_row):
        try:
            for column_name in self.additional_drug_columns:
                value = first_row.get(column_name)

                if pd.notna(value) and str(value).strip() == '1':
                    drug_name = column_name.replace('has_', '').replace('_', ' ').title()
                    drug = AdditionalDrugs(
                        patient_id=str(patient_id),
                        name=to_python_format(drug_name)
                    )
                    self.session.add(drug)
        except Exception as e:
            raise TypeError(f"[ERROR] import_additional_drugs for ID {patient_id}: {e}")

    def _import_comorbidities(self, patient_id, first_row):
        try:

            for col_name in self.comorbidities_columns:
                value = first_row.get(col_name)

                if pd.notna(value) and str(value).strip() == '1':
                    concomitant_name = col_name.replace('has_', '').replace('_', ' ').title()
                    concomitant = Comorbidities(
                        patient_id=str(patient_id),
                        name=to_python_format(concomitant_name)
                    )
                    self.session.add(concomitant)
        except Exception as e:
            raise TypeError(f"[ERROR] import_additional_drugs for ID {patient_id}: {e}")

    def _import_dietary_intake(self, patient_id, group):
        try:
            for index, row in group.iterrows():
                value = row.get('Dietary intake')
                if pd.notna(value) and str(value).strip() == '1':
                    time_fields = self._extract_time_fields(row)

                    dietary = DietaryIntake(
                        patient_id=str(patient_id),
                        **time_fields
                    )
                    self.session.add(dietary)
        except Exception as e:
            raise TypeError(f"[ERROR] import_dietary_intake for ID {patient_id}: {e}")

    def _import_measurement(self, patient_id, group):
        try:
            for _, row in group.iterrows():
                cgm = row.get('CGM (mg / dl)')
                cbg = row.get('CBG (mg / dl)')
                blood_ketone = row.get('Blood Ketone (mmol / L)')
                time_fields = self._extract_time_fields(row)

                measurement = Measurement(
                    patient_id=str(patient_id),
                    **time_fields,
                    cgm=to_python_format(cgm) if pd.notna(cgm) else 0.0,
                    cbg=to_python_format(cbg) if pd.notna(cbg) else 0.0,
                    blood_ketone=to_python_format(blood_ketone) if pd.notna(blood_ketone) else 0.0,

                )
                self.session.add(measurement)

        except Exception as e:
            raise TypeError(f'[ERROR] _import_measurement for ID {patient_id}: {e}')


    def _import_insulin_dose(self, patient_id, group):
        try:

            for _, row in group.iterrows():
                time_fields = self._extract_time_fields(row)

                for col in self.insulin_columns:
                    dose_value = row.get(col)
                    if pd.notna(dose_value) and float(dose_value) > 0:
                        insulin_name = col.replace("dose_", "").replace("_", " ").title()
                        insulin_id = self._get_or_create_insulin_id(insulin_name)

                        taking = TakingInsulin(
                            patient_id=str(patient_id),
                            insulin_id=insulin_id,
                            dose=to_python_format(float(dose_value)),
                            **time_fields
                        )
                        self.session.add(taking)

        except Exception as e:
            self.session.rollback()
            raise TypeError(f'[ERROR] import_insulin_doses for ID {patient_id}: {e}')


    def _import_diabetes_tablet(self, patient_id, group):
        try:
            diabetes_tablet_cols = [
                'dose_dapagliflozin', 'dose_metformin', 'dose_sitagliptinphosphate_metforminhydrochloride',
                'dose_voglibose', 'dose_repaglinide', 'dose_gliclazide', 'dose_acarbose', 'dose_liraglutide',
                'dose_sitagliptin', 'dose_gliquidone', 'dose_canagliflozin', 'dose_pioglitazone',
                'dose_glimepiride', 'dose_empagliflozin', 'dose_linagliptin'
            ]

            for _, row in group.iterrows():
                time_fields = self._extract_time_fields(row)

                for col in diabetes_tablet_cols:
                    dose_value = row.get(col)

                    if pd.notna(dose_value) and float(dose_value) > 0:
                        diabetes_tablet = col.replace("dose_", "").replace("_", " ").title()
                        diabetes_tablet_id = self._get_or_create_diabetes_tablet(diabetes_tablet)

                        taking = TakingDiabetesTablet(
                            patient_id=str(patient_id),
                            diabetes_tablet_id=diabetes_tablet_id,
                            dose=to_python_format(float(dose_value)),
                            **time_fields
                        )
                        self.session.add(taking)

        except Exception as e:
            raise TypeError(f'[ERROR] _import_diabetes_tablet for ID {patient_id}: {e}')

    def _import_therapy(self, patient_id, group):
        pass

    def _get_or_create_partition(self, partition_name: str):
        partition = self.session.query(DatasetPartition).filter_by(name=partition_name).first()
        if not partition:
            partition = DatasetPartition(name=partition_name)
            self.session.add(partition)
            self.session.flush()  # get id before commit
        return partition


    def _get_or_create_insulin_id(self, name: str) -> int:
        insulin = self.session.query(Insulin).filter_by(name=name).first()

        if not insulin:
            insulin = Insulin(name=name)
            self.session.add(insulin)
            self.session.flush()

        return insulin.id


    def _get_or_create_diabetes_tablet(self, name: str) -> int:
        diabetes_tablet = self.session.query(DiabetesTablets).filter_by(name=name).first()

        if not diabetes_tablet:
            diabetes_tablet = DiabetesTablets(name=name)
            self.session.add(diabetes_tablet)
            self.session.flush()

        return diabetes_tablet.id
