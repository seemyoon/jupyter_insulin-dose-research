from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship

from diabetes_project.infrastructure.db.base import Base


class TakingInsulin(Base):
    __tablename__ = 'taking_insulin'

    id = Column(Integer, primary_key=True)
    patient_id = Column(String, ForeignKey("patient.id"))
    insulin_id = Column(Integer, ForeignKey("insulin.id"))

    patient = relationship("Patient", back_populates="insulins")
    insulin = relationship("Insulin", back_populates="taking_insulins")

    dose = Column(Float)
