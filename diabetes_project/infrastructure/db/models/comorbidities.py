from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from diabetes_project.infrastructure.db.base import Base


class Comorbidities(Base):
    __tablename__ = "comorbidities"

    id = Column(Integer, primary_key=True)
    patient_id = Column(String, ForeignKey("patient.id"))

    patient = relationship("Patient", back_populates="comorbidities")

    name = Column(String)
