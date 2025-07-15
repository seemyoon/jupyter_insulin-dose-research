from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from db.base import Base


class Comorbidities(Base):
    __tablename__ = "comorbidities"

    id = Column(Integer, primary_key=True, nullable=False)
    patient_id = Column(String, ForeignKey("patient.id"), nullable=False)

    patient = relationship("Patient", back_populates="comorbidities")

    name = Column(String)
