import os
import time
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

BASE_DIR = Path(__file__).resolve().parent.parent

dotenv_path = BASE_DIR / '.env'

load_dotenv(dotenv_path)

DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def connect_to_db(db_url, wait_seconds=3):
    while True:
        try:
            engine = create_engine(db_url)
            with engine.connect():
                print('connection to db was success')
                return engine
        except OperationalError as e:
            print(f'error connection {e}')
            print(f"repeating after {wait_seconds} seconds...")
            time.sleep(wait_seconds)

engine = connect_to_db(DATABASE_URL)