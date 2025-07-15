from db.engine import engine
from db.base import Base
from db.models import *

Base.metadata.create_all(bind=engine)

print("tables were successfully created ")
