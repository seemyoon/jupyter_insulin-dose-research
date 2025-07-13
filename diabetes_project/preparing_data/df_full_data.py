import re
from pathlib import Path

import pandas as pd

from df_shanghai_summary import main
from df_shanghai_time_series import DfShanghaiTimeSeries


class DfFullData:
    def __init__(self, folder_path):
        self.df = None
        self.folder_path = Path(folder_path)
        self._load_and_merge_final_data()

    def _load_and_merge_final_data(self):
        merger = DfShanghaiTimeSeries(self.folder_path)
        merger_df = merger.merge_all_data_ts()

        summary_df = main()

        summary_ids = set(summary_df['Patient Number'].unique())
        merger_df = merger_df[merger_df['Patient Number'].isin(summary_ids)]

        merger_df = pd.merge(merger_df, summary_df, on='Patient Number', how='left')

        merger_df['Date'] = pd.to_datetime(merger_df['Date'], format='mixed', errors='coerce')
        merger_df['minute_treat'] = merger_df['Date'].dt.minute
        merger_df['hour_of_day_treat'] = merger_df['Date'].dt.hour
        merger_df['month_treat'] = merger_df['Date'].dt.month
        merger_df['day_treat'] = merger_df['Date'].dt.day
        merger_df['year_treat'] = merger_df['Date'].dt.year

        merger_df = merger_df.drop(columns=['Date'])

        first_cols = ['Patient Number', 'year_treat', 'month_treat', 'day_treat', 'hour_of_day_treat', 'minute_treat']
        others_cols = [col for col in merger_df.columns if col not in first_cols]
        merger_df = merger_df[first_cols + others_cols]
        extra_ids = set(merger_df['Patient Number'].unique()) - summary_ids

        if extra_ids: print('Extra IDs (in merger_df but not in summary_df):', extra_ids)

        self.df = merger_df
        return merger_df

    def process_col(self, name_col, match_pattern, split_by):
        unique_values = set()

        for values in self._load_and_merge_final_data()[name_col].dropna():
            parts = values.split(split_by)

            for part in parts:
                match = re.match(match_pattern, part.strip())

                if match:
                    unique_values.add(match.group(1).strip())
                else:
                    print(f'missed matches in column: {name_col}', part)

        return unique_values

    def __normalize_insulin_dose_sc(self):

        insulin_dose_sc = self.process_col('Insulin dose - s.c.', r'\s*(.*?),\s*\d+\s*IU', ';')

        med_normalization_map = {
            'insulin\xa0glargine': 'insulin_glargine',
            'insulin glarigine': 'insulin_glargine',
            'insulin glargine': 'insulin_glargine',
            'insulin detemir': 'insulin_detemir',
            'insulin degludec': 'insulin_degludec',
            'insulin aspart': 'insulin_aspart',
            'insulin aspart 70/30': 'insulin_aspart_70_30',
            'insulin glulisine': 'insulin_glulisine',
            'SciLin M30': 'scilin_m30',
            'Humulin R': 'humulin_r',
            'Humulin 70/30': 'humulin_70_30',
            'Novolin 30R': 'novolin_30r',
            'Novolin 50R': 'novolin_50r',
            'Novolin R': 'novolin_r',
            'Gansulin R': 'gansulin_r',
            'Gansulin 40R': 'gansulin_40r',
        }

        for medicament in insulin_dose_sc:
            norm_med = med_normalization_map.get(medicament.strip(),
                                                 medicament.strip().lower().replace(" ", "_").replace("-", "_"))
            col_name = f'dose_{norm_med}'
            """
            ➤ Suppose x is "Humulin R, 2 IU; insulin degludec, 12 IU". medicament - "insulin degludec"
            
            ➤ x.split(‘;’) - ["Humulin R, 2 IU", " insulin degludec, 12 IU"]
            
            ➤ for part in x.split(‘;’) - first "Humulin R, 2 IU" and then " insulin degludec, 12 IU"
            
            ➤ if part.strip().lower().startswith(medicament.lower()) - If part of the string starts with the name of the medicine (medicament) - ignoring spaces and case - then process that part. " insulin degludec, 12 IU" - it starts with "insulin degludec".
            
            ➤ part.strip().split(‘,’)[1]: → "insulin degludec, 12 IU" → split(‘,’) → ["insulin degludec", " 12 IU"] → Take [1] → " 12 IU"
            
            ➤ .strip().split()[0] → " 12 IU" → .strip() → "12 IU" → .split() → ["12", "IU"] → Take [0] → "12"
            
            ➤ int(...)→ Convert "12" → 12 (int type)
            """
            self.df[col_name] = self.df['Insulin dose - s.c.'].apply(
                lambda x: sum(
                    int(part.strip().split(',')[1].strip().split()[0])
                    for part in x.split(';')
                    if part.strip().lower().startswith(medicament.lower())
                ) if pd.notna(x) else 0
            )
        self.df = self.df.drop(columns=['Insulin dose - s.c.'], axis=1)

        return self

    def __normalize_insulin_dose_iv(self):
        insulin_names = set()

        for text in self.df[
            'Insulin dose - i.v.'].dropna():  # search all rows from the ‘Insulin dose - i.v.’ column that are not empty
            matches = re.findall(r'(\d+)\s*IU\s+([A-Za-z][A-Za-z\s\d\-]*)',
                                 text)  # regular expression that searches for fragments of the following type: 12 IU Novolin R

            # In each line, look for all fragments that match the pattern (number + IU + name)
            # For example, if the string is: 500ml 0.9% sodium chloride, 12 IU Novolin R, 10 ml KCl The matches would be: [(‘12’, ‘Novolin R’)].

            for dose, name in matches:
                insulin_names.add(name.strip())
                # Add the found insulin name (e.g. Novolin R) to the insulin_names set.

        for medicament in insulin_names:
            norm_med = medicament.strip().lower().replace(" ", "_").replace("-", "_")
            col_name = f'dose_{norm_med}'

            new_values = self.df['Insulin dose - i.v.'].apply(
                lambda x: sum(
                    int(match.group(1))
                    for match in re.finditer(rf'(\d+)\s*IU\s+{re.escape(medicament)}', x, re.IGNORECASE)
                ) if pd.notna(x) else 0
            )

            existing_col = self.df.get(col_name, pd.Series(0, index=self.df.index))
            self.df[col_name] = existing_col + new_values

        self.df = self.df.drop(columns=['Insulin dose - i.v.'], axis=1)

        return self

    def __normalize_csii_dose_insulin(self):
        cols = [
            'CSII - bolus insulin (Novolin R, IU)',
            'CSII - basal insulin (Novolin R, IU / H)',
        ]

        for col in cols:
            def extract_dose(value):
                if isinstance(value, str) and "temporarily suspend insulin delivery" in value.lower():
                    return 0
                try:
                    return float(value)
                except:
                    return 0

            existing = self.df.get('dose_novolin_r', pd.Series(0, index=self.df.index)).fillna(0)

            doses = self.df[col].apply(extract_dose).fillna(0)
            existing = existing + doses

            self.df['dose_novolin_r'] = existing
            self.df = self.df.drop(columns=[col], axis=1)

        return self

    def __normalize_non_insulin_agents(self):
        def dose_to_mg(dose_str: str) -> float:
            match = re.match(r'(\d+(?:\.\d+)?)\s*(mg|g)', dose_str.strip())
            if not match:
                return 0.0
            value, unit = float(match.group(1)), match.group(2).lower()

            if unit == 'g':
                return value * 1000
            return value

        unique_values = set()

        for values in self.df['Non-insulin hypoglycemic agents'].dropna():
            parts = [part.strip() for part in values.split(',')]

            for part in parts:
                part_clean = part.lower().replace(" tablets", "").replace("/", "_").strip()
                match = re.match(
                    r'([a-z\d_\- ]+?)\s+((?:\d+(?:\.\d+)?\s*(?:mg|g))(?:\s*/\s*\d+(?:\.\d+)?\s*(?:mg|g))?)', part_clean)

                if match:
                    substance = match.group(1).strip()
                    dose_str = match.group(2).strip()
                    unique_values.add((substance, dose_str))
                else:
                    print(f"missed matches in column: 'Non-insulin hypoglycemic agents'", part)

        for substance, _ in unique_values:
            col_name = f'dose_{substance.replace(" ", "_")}'
            if col_name not in self.df.columns:
                self.df[col_name] = 0.0

        def parse_and_sum_doses(cell_value):
            doses = {}
            if pd.isna(cell_value):
                return doses

            parts = [part.strip() for part in cell_value.lower().replace(" tablets", "").replace("/", "_").split(',')]
            for part in parts:
                match = re.match(
                    r'([a-z\d_\- ]+?)\s+((?:\d+(?:\.\d+)?\s*(?:mg|g))(?:\s*/\s*\d+(?:\.\d+)?\s*(?:mg|g))?)', part)

                if not match:
                    continue

                substance = match.group(1).strip()
                dose_str = match.group(2).strip()

                dose_parts = [d.strip() for d in dose_str.split('/')]
                dose_mg_total = sum(dose_to_mg(d) for d in dose_parts)

                doses[substance] = doses.get(substance, 0) + dose_mg_total
            return doses

        for idx, row in self.df.iterrows():
            doses = parse_and_sum_doses(row['Non-insulin hypoglycemic agents'])
            for substance, dose_val in doses.items():
                col_name = f'dose_{substance.replace(" ", "_")}'
                if col_name in self.df.columns:
                    self.df.at[idx, col_name] += dose_val
        self.df = self.df.drop(columns=['Non-insulin hypoglycemic agents'], axis=1)

        return self

    def __fill_missing_values(self):
        mask = self.df['Patient Number'] == '2029_0_20210526'

        mean_cgm = self.df.loc[mask, 'CGM (mg / dl)'].mean()

        self.df.loc[mask, 'CGM (mg / dl)'] = self.df.loc[mask, 'CGM (mg / dl)'].fillna(mean_cgm)

        self.df["CBG (mg / dl)"] = self.df["CBG (mg / dl)"].fillna(0)
        self.df["Blood Ketone (mmol / L)"] = self.df['Blood Ketone (mmol / L)'].fillna(0)

        return self

    def normalize_all(self):
        return (self
                .__normalize_insulin_dose_sc()
                .__normalize_insulin_dose_iv()
                .__normalize_csii_dose_insulin()
                .__normalize_non_insulin_agents()
                .__fill_missing_values())
