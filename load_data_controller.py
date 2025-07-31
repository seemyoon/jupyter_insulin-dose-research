from pathlib import Path

from load_data_to_db import DataImporter


def main():
    root_dir = Path(__file__).resolve().parent

    data = {
        'train': {
            'name': 'train',
            'file': root_dir / "cleaned_data" / "train_data.csv"
        },
        'test': {
            'name': 'test',
            'file': root_dir / "cleaned_data" / "test_data.csv"
        },

    }

    if not data['train']['file'].exists():
        print(f"file is not exist: {data['train']['name']}")
        return

    if not data['test']['file'].exists():
        print(f"file is not exist: {data['test']['name']}")
        return

    importer = DataImporter(data['train']['name'])
    importer.import_from_data(str(data['train']['file']))


if __name__ == '__main__': main()
