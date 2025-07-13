from sqlalchemy.orm import declarative_base
from diabetes_project.infrastructure.db.engine import engine

Base = declarative_base()

Base.metadata.create_all(engine)