from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship

from db.base import Base


class TakingInsulin(Base):
    __tablename__ = 'taking_insulin'

    id = Column(Integer, primary_key=True, nullable=False)
    patient_id = Column(String, ForeignKey("patient.id"), nullable=False)
    insulin_id = Column(Integer, ForeignKey("insulin.id"), nullable=False)

    patient = relationship("Patient", back_populates="insulins")
    insulin = relationship("Insulin", back_populates="taking_insulins")

    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    hour = Column(Integer)
    minute = Column(Integer)

    dose = Column(Float)
