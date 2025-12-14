from pathlib import Path

from load_data_to_db import DataImporter


def main():
    root_dir = Path(__file__).resolve().parent

    data_file_path = root_dir / "data" / "Shanghai_data.csv"

    if not data_file_path.exists():
        print(f"file is not exist: {data_file_path}")
        return

    importer = DataImporter()
    importer.import_from_data(str(data_file_path))


if __name__ == '__main__': main()
