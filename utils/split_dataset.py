import os

import pandas as pd


def split_dataset(df: pd.DataFrame, test_ids):
    """

    :param df: input Dataframe with column 'Patient Number'
    :param test_ids: List of Patient Number, that should be included in test
    :return: (train_df, test_df)
    """

    if 'Patient Number' not in df.columns:
        raise ValueError("Dataframe must include column 'Patient Number'.")

    test_df = df[df['Patient Number'].isin(test_ids)].copy()
    train_df = df[~df['Patient Number'].isin(test_ids)].copy()

    return test_df, train_df


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_PATH = os.path.join(CURRENT_DIR, '..', 'cleaned_data', 'finish_data.csv')

print("File exists?", os.path.exists(FILE_PATH))
df = pd.read_csv(FILE_PATH)

test_ids = ['1011_0_20210622', '1009_0_20210803', '1012_0_20210923', '1010_0_20210915', '1003_0_20210831',
            '2055_0_20210524', '2014_0_20201224', '2014_1_20210317', '2009_0_20211103',
            '2056_0_20210524', '2016_0_20201224', '2012_0_20220126', '2053_0_20210518', '2015_0_20210203',
            '2011_0_20220123', '2010_0_20220111', '2015_1_20210219', '2052_0_20210511',
            '2055_1_20201207', '2013_0_20220123', '2002_0_20210513']

train_df, test_df = split_dataset(df, test_ids)

train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
