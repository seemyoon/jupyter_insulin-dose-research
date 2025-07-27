from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship

from db.base import Base


class Hospitalization(Base):
    __tablename__ = 'hospitalization'

    id = Column(Integer, primary_key=True, nullable=False)

    patient_id = Column(String, ForeignKey('patient.id'), nullable=False)
    insulin_id = Column(Integer, ForeignKey('insulin.id'), nullable=False)

    patient = relationship("Patient", back_populates="hospitalizations")
    insulin = relationship('Insulin', back_populates="hospitalizations")

    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    hour = Column(Integer)
    minute = Column(Integer)

    dose = Column(Float)
