from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from db.base import Base


class PatientAdditionalDrug(Base):
    __tablename__ = 'patient_additional_drug'

    id = Column(Integer, primary_key=True)
    patient_id = Column(String, ForeignKey('patient.id'), nullable=False)
    drug_id = Column(Integer, ForeignKey('additional_drug.id'), nullable=False)

    patient = relationship('Patient', back_populates='additional_drugs')
    drug = relationship('AdditionalDrug', back_populates='patient')