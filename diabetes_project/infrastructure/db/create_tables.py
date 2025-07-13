from diabetes_project.infrastructure.db.engine import engine
from diabetes_project.infrastructure.db.base import Base
from diabetes_project.infrastructure.db.models import *

Base.metadata.create_all(bind=engine)

print("tables were successfully created ")
