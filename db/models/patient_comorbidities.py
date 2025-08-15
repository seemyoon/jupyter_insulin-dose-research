from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from db.base import Base


class PatientComorbidities(Base):
    __tablename__ = 'patient_comorbidities'

    id = Column(Integer, primary_key=True)
    patient_id = Column(String, ForeignKey('patient.id'), nullable=False)
    comorbidity_id = Column(Integer, ForeignKey('comorbidities.id'), nullable=False)

    patient = relationship('Patient', back_populates='comorbidities')
    comorbidity = relationship('Comorbidities', back_populates='patient')