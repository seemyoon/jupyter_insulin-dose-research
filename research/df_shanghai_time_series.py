from pathlib import Path
import pandas as pd


class DfShanghaiTimeSeries:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.combined_data = None

    def merge_all_data_ts(self):
        folders = ['T1DM', 'T2DM']
        all_data = []

        for folder in folders:
            folder_path = self.folder_path / folder

            if not folder_path.exists():
                print(f"folder not found: {folder_path}")

            for file in folder_path.glob('*.csv'):
                try:
                    df = pd.read_csv(file)

                    df['Patient Number'] = file.stem

                    df.rename(columns={
                        'CGM ': 'CGM (mg / dl)',
                        'CBG ': 'CBG (mg / dl)',

                        'Blood Ketone ': 'Blood Ketone (mmol / L)',
                        'CSII - bolus insulin (Novolin R  IU)': 'CSII - bolus insulin (Novolin R, IU)',
                        'CSII - bolus insulin': 'CSII - bolus insulin (Novolin R, IU)',

                        'CSII - basal insulin (Novolin R  IU / H)': 'CSII - basal insulin (Novolin R, IU / H)',
                        '胰岛素泵基础量 (Novolin R, IU / H)': 'CSII - basal insulin (Novolin R, IU / H)',
                        'CSII - basal insulin': 'CSII - basal insulin (Novolin R, IU / H)',
                    }, inplace=True)

                    all_data.append(df)
                except Exception as e:
                    print(f'error reading {file}: {e}')

        if not all_data:
            print("no data loaded")
            return None

        combined_df = pd.concat(all_data, ignore_index=True)

        combined_df['Dietary intake'] = combined_df['Dietary intake'].notna().astype(int)

        combined_df.columns = [col.strip() for col in combined_df.columns]

        combined_df.drop(columns=[
            '饮食',
            # 'Date'
            '进食量',
            'CSII - bolus insulin',
            'CSII - basal insulin'
        ], inplace=True)

        self.combined_data = combined_df
        return self.combined_data
