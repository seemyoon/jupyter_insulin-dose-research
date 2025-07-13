import os
from pathlib import Path
import pandas as pd
from sklearn import preprocessing


class DiabetesDataPreprocessor:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.df = None
        self._validate_path()

    def _validate_path(self):
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Directory {self.folder_path} doesn't exist")

    def load_and_combine_data(self):
        """Load and combine T1DM and T2DM datasets"""
        df1 = pd.read_csv(self.folder_path.joinpath('Shanghai_T1DM_Summary.csv'))
        df2 = pd.read_csv(self.folder_path.joinpath('Shanghai_T2DM_Summary.csv'))

        self.df = pd.concat([df1, df2], ignore_index=True)
        return self

    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        self.df.replace('/', pd.NA, inplace=True)

        cols_to_fill = [
            'Age (years)', 'Height (m)', 'Weight (kg)', 'BMI (kg/m2)',
            'Smoking History (pack year)', 'Duration of Diabetes (years)',
            'Fasting Plasma Glucose (mg/dl)', '2-hour Postprandial Plasma Glucose (mg/dl)',
            'Fasting C-peptide (nmol/L)', '2-hour Postprandial C-peptide (nmol/L)',
            'Fasting Insulin (pmol/L)', '2-hour Postprandial Insulin (pmol/L)',
            'HbA1c (mmol/mol)', 'Glycated Albumin (%)', 'Total Cholesterol (mmol/L)',
            'Triglyceride (mmol/L)', 'High-Density Lipoprotein Cholesterol (mmol/L)',
            'Low-Density Lipoprotein Cholesterol (mmol/L)', 'Creatinine (umol/L)',
            'Estimated Glomerular Filtration Rate  (ml/min/1.73m2)',
            'Uric Acid (mmol/L)', 'Blood Urea Nitrogen (mmol/L)'
        ]

        for col in cols_to_fill:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(
                    self.df[col].astype(str).str.strip(),
                    errors='coerce'
                )

        self.df[cols_to_fill] = self.df[cols_to_fill].fillna(self.df[cols_to_fill].median())
        return self

    def clean_data(self):
        """Remove outliers and fix data errors"""
        self.df = self.df[self.df['Fasting Insulin (pmol/L)'] < 700]
        self.df = self.df[self.df['2-hour Postprandial Insulin (pmol/L)'] < 800]

        # Fix specific errors in Other Agents column
        fixes = {
            'raberazole': 'rabeprazole',
            'calcium carbonate and vitamin D3 tablet': 'calcium carbonate, vitamin D3 tablet',
            'rosuvastatinqn': 'rosuvastatin',
            'nifedipine doxazosin': 'nifedipine, doxazosin'
        }

        for wrong, correct in fixes.items():
            self.df['Other Agents'] = self.df['Other Agents'].str.replace(wrong, correct)

        return self

    def add_group_flags(self, column_name, items_to_group):
        """Create binary flags for grouped items"""
        all_groups = sorted(set(items_to_group.values()))

        def parse_agents(row):
            if pd.isna(row): return []
            return [agent.strip() for agent in row.split(',')]

        for group in all_groups:
            self.df[f'has_{group}'] = self.df[column_name].apply(
                lambda x: int(any(items_to_group.get(agent) == group for agent in parse_agents(x)))
            )

        self.df = self.df.drop(columns=[column_name], axis=1)
        return self

    def specify_has_or_no(self, column_name):
        """Create binary flag for any value in column"""
        column_name = column_name.strip()

        def has_any_value(row):
            if pd.isna(row): return 0
            for el in row.split(','):
                el = el.strip().lower()
                if el and el != 'none':
                    return 1
            return 0

        new_name_column = f'has_{column_name.lower().replace(" ", "_")}'
        self.df[new_name_column] = self.df[column_name].apply(has_any_value)
        self.df = self.df.drop(columns=[column_name])
        return self

    def encode_categorical(self, columns):
        """Label encode categorical columns"""
        label_encoder = preprocessing.LabelEncoder()
        for col in columns:
            self.df[col] = label_encoder.fit_transform(self.df[col])
        return self

    def rename_column(self, old_name, new_name):
        """Rename a column and drop the old one"""
        self.df[new_name] = self.df[old_name]
        self.df = self.df.drop(columns=[old_name])
        return self

    def drop_columns(self, columns):
        """Drop specified columns"""
        self.df = self.df.drop(columns=columns)
        return self

    def get_data(self):
        """Return the processed dataframe"""
        return self.df


class DiabetesFeatureEngineer:
    @staticmethod
    def find_unknown_agents(df, col_name, items_to_group):
        """Find unknown agents not in the grouping dictionary"""
        all_agents = set()

        for elem in df[col_name]:
            agents = [agent.strip() for agent in elem.split(',')]
            all_agents.update(agents)

        unknown_agents = sorted(
            [unk_ag for unk_ag in all_agents
             if unk_ag not in items_to_group and unk_ag.lower() != 'none']
        )

        return unknown_agents

    @staticmethod
    def get_drug_groupings():
        """Return dictionary for drug groupings"""
        return {
            # hypolipidemic
            'pravastatin': 'hypolipidemic',
            'rosuvastatin': 'hypolipidemic',
            'fenofibrate': 'hypolipidemic',
            'ezetimibe': 'hypolipidemic',
            'atorvastatin': 'hypolipidemic',

            # angioprotectors
            'calcium dobesilate': 'angioprotectors',
            'beiprostaglandin sodium': 'angioprotectors',

            # ace inhibitors
            'benazepril': 'ace_inhibitors',

            # minerals and vitamins
            'potassium chloride': 'minerals_and_vitamins',
            'calcium carbonate': 'minerals_and_vitamins',
            'calcitriol': 'minerals_and_vitamins',
            'multivitamin': 'minerals_and_vitamins',
            'vitamin B1': 'minerals_and_vitamins',
            'vitamin D3 tablet': 'minerals_and_vitamins',
            'mecobalamin': 'minerals_and_vitamins',

            # probiotics
            'clostridium butyricum': 'probiotics',

            # ARB
            'telmisartan': 'arb',
            'valsartan': 'arb',
            'olmesartan medoxomil': 'arb',
            'olmesartan': 'arb',
            'losartan': 'arb',
            'losartan/hydrochlorothiazide': 'arb',
            'irbesartan': 'arb',
            'candesartan': 'arb',
            'allisartan': 'arb',

            # psychotropic
            'quetiapine': 'psychotropic',

            # antianginal
            'isosorbide mononitrate': 'antianginal',

            # gout treatment
            'febuxostat': 'gout_treatment',

            # laxatives
            'bisacodyl': 'laxatives',

            # urological drugs
            'Qianlie Shutong capsule  (Chinese patent drug for prostatic hyperplasia)': 'urological_drugs',

            # calcium channel blockers
            'nifedipine': 'calcium_channel_blockers',
            'amlodipine': 'calcium_channel_blockers',
            'felodipine': 'calcium_channel_blockers',
            'benidipine': 'calcium_channel_blockers',

            # antiarrhythmic
            'doxazosin': 'antiarrhythmic',
            'labetalol': 'antiarrhythmic',
            'bisoprolol': 'antiarrhythmic',
            'metoprolol': 'antiarrhythmic',

            # gastroprotective
            'rabeprazole': 'gastroprotective',

            # circulatory support
            'Yinxingye tablet (extract of Ginkgo biloba leaves)': 'circulatory_support',

            # antithrombotic
            'aspirin': 'antithrombotic',
            'clopidogrel': 'antithrombotic',
            'rivaroxaban': 'antithrombotic',

            # vasodilators
            'trimetazidine': 'vasodilators',
            'magnesium isoglycyrrhizinate': 'vasodilators',

            # pancreatic
            'pancreatic kininogenase': 'pancreatic',

            # neuroprotectors
            'epalrestat': 'neuroprotectors',

            # kidney support
            'compound α-keto acid tablet': 'kidney_support',
            'Shen Shuai Ning capsule (Chinese patent drug for renal dysfunction)': 'kidney_support',

            # hepatoprotector
            'polyene phosphatidylcholine': 'hepatoprotector',
            'diammonium glycyrrhizinate': 'hepatoprotector',

            # immunomodulators
            'leucogen': 'immunomodulators',

            # thyroid diseases
            'levothyroxine': 'thyroid_diseases',

            # antibiotics
            'levofloxacin': 'antibiotics',

            # antihypertensives
            'Zhenju Jiangya tablet (Chinese patent drug for hypertension)': 'antihypertensives',

            # vestibular disorders
            'betahistine': 'vestibular_disorders',
        }

    @staticmethod
    def get_disease_groupings():
        """Return dictionary for disease groupings"""
        return {
            # diseases_of_the_stomach_and_intestines
            'chronic atrophic gastritis': 'diseases_of_the_stomach_and_intestines',
            'colorectal polyp': 'diseases_of_the_stomach_and_intestines',
            'chronic gastritis': 'diseases_of_the_stomach_and_intestines',
            'gastric polyp': 'diseases_of_the_stomach_and_intestines',

            # diseases_of_the_musculoskeletal_system
            'lumbar herniated disc': 'diseases_of_the_musculoskeletal_system',
            'osteopenia': 'diseases_of_the_musculoskeletal_system',
            'osteoporosis': 'diseases_of_the_musculoskeletal_system',
            'lumbar spine tumor': 'diseases_of_the_musculoskeletal_system',

            # cardiovascular_diseases
            'myocardial bridging': 'cardiovascular_diseases',
            'sinus arrhythmia': 'cardiovascular_diseases',
            'hypertension': 'cardiovascular_diseases',
            'hyperlipidemia': 'cardiovascular_diseases',
            'sinus bradycardia': 'cardiovascular_diseases',
            'atrial fibrillation': 'cardiovascular_diseases',

            # kidney_diseases
            'kidney cyst': 'kidney_diseases',
            'hydronephrosis': 'kidney_diseases',
            'nephrolithiasis': 'kidney_diseases',
            'urinary tract infection': 'kidney_diseases',

            # dental_diseases
            'periodontitis': 'dental_diseases',

            # gynecological_diseases
            'hysteromyoma': 'gynecological_diseases',

            # neurological_and_psychiatric_diseases
            'anxiety': 'neurological_and_psychiatric_diseases',
            'cerebrovascular disease': 'neurological_and_psychiatric_diseases',
            "Alzheimer's disease": 'neurological_and_psychiatric_diseases',
            "Parkinson's disease": 'neurological_and_psychiatric_diseases',

            # liver_diseases
            'fatty liver disease': 'liver_diseases',
            'fatty liver disese': 'liver_diseases',
            'liver cyst': 'liver_diseases',
            'hepatic dysfunction': 'liver_diseases',

            # gallbladder_diseases
            'cholecystitis': 'gallbladder_diseases',
            'cholelithiasis': 'gallbladder_diseases',
            'gallbladder polyp': 'gallbladder_diseases',

            # infectious_diseases
            'chronic hepatitis B': 'infectious_diseases',

            # oncology
            'breast cancer': 'oncology',
            'pancreatic cancer': 'oncology',
            'parotid gland carcinoma': 'oncology',
            'lung lesion': 'oncology',
            'pulmonary nodule': 'oncology',

            # endocrine_diseases
            'hypoparathyroidism': 'endocrine_diseases',
            'hypothyroidism': 'endocrine_diseases',
            'enlarged adrenal gland': 'endocrine_diseases',
            'thyroid nodule': 'endocrine_diseases',

            # male_reproductive_diseases
            'prostatic hyperplasia': 'male_reproductive_diseases',

            # eye_diseases
            'cataract': 'eye_diseases',
            'conjunctivitis': 'eye_diseases',

            # hematologic_disorders
            'hypoleukocytemia': 'hematologic_disorders',
            'leucopenia': 'hematologic_disorders',

            # autoimmune_diseases
            'systemic sclerosis': 'autoimmune_diseases',
            'psoriasis': 'autoimmune_diseases',

            # electrolyte_and_mineral_disorders
            'hypocalcemia': 'electrolyte_and_mineral_disorders',
            'hypokalemia': 'electrolyte_and_mineral_disorders',
            'vitamin D deficiency': 'electrolyte_and_mineral_disorders',
            'hyperuricemia': 'electrolyte_and_mineral_disorders',
        }


def main():
    # Initialize preprocessor
    preprocessor = DiabetesDataPreprocessor('../data/Shanghai_diabetes_datasets/clinical_info/csv')

    # Load and process data
    df = (preprocessor
          .load_and_combine_data()
          .handle_missing_values()
          .clean_data()
          .add_group_flags('Other Agents', DiabetesFeatureEngineer.get_drug_groupings())
          .add_group_flags('Comorbidities', DiabetesFeatureEngineer.get_disease_groupings())
          .drop_columns(['Hypoglycemic Agents'])
          .specify_has_or_no('Diabetic Microvascular Complications')
          .specify_has_or_no('Diabetic Macrovascular  Complications')
          .specify_has_or_no('Acute Diabetic Complications')
          .encode_categorical([
        'Alcohol Drinking History (drinker/non-drinker)',
        'Hypoglycemia (yes/no)',
        'Type of Diabetes'
    ])
          .rename_column('Hypoglycemia (yes/no)', 'has_hypoglycemia')
          .get_data())


    print(f"Final dataframe shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = main()
