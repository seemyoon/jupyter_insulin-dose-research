from sqlalchemy.orm import declarative_base
from db.engine import engine

Base = declarative_base()

Base.metadata.create_all(engine)