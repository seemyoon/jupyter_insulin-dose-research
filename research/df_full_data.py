import re
from pathlib import Path

import pandas as pd

from df_shanghai_summary import main
from df_shanghai_time_series import DfShanghaiTimeSeries


class DfFullData:
    def __init__(self, folder_path):
        self.df = None
        self.folder_path = Path(folder_path)

    def get_final_data(self):
        merger = DfShanghaiTimeSeries(self.folder_path)
        # '../data/Shanghai_diabetes_datasets/Shanghai_CSV-Data'
        merger_df = merger.merge_all_data_ts()

        summary_df = main()

        merger_df = pd.merge(merger_df, summary_df, on='Patient Number', how='left')

        # summary_df.to_csv('summary_df.csv')
        # merger_df.to_csv('merger_df.csv')


        # merger_df['minute_treat'] = merger_df['Date'].dt.minute
        # merger_df['hour_of_day_treat'] = merger_df['Date'].dt.hour
        # merger_df['month_treat'] = merger_df['Date'].dt.month
        # merger_df['year_treat'] = merger_df['Date'].dt.year
        # merger_df['day_treat'] = merger_df['Date'].dt.year

        summary_ids = set(summary_df['Patient Number'].unique())
        merger_ids = set(merger_df['Patient Number'].unique())

        missing_ids = summary_ids - merger_ids
        extra_ids = merger_ids - summary_ids

        if missing_ids:
            print('Missing IDs (in summary_df but not in merger_df):', missing_ids)

        if extra_ids:
            print('Extra IDs (in merger_df but not in summary_df):', extra_ids)

        self.df = merger_df

        return self.df

    def process_col(self, name_col, match_pattern, split_by):
        unique_values = set()

        for values in self.get_final_data()[name_col].dropna():
            parts = values.split(split_by)

            for part in parts:
                match = re.match(match_pattern, part.strip())

                if match:
                    unique_values.add(match.group(1).strip())
                else:
                    print(f'missed matches in column: {name_col}', part)
