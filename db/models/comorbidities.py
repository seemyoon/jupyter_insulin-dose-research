from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from db.base import Base


class Comorbidities(Base):
    __tablename__ = "comorbidities"

    id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String)

    patient = relationship("PatientComorbidities", back_populates="comorbidity")