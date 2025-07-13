from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship

from diabetes_project.infrastructure.db.base import Base


class PatientMedicalStatic(Base):
    __tablename__ = 'medical_static'

    id =Column(Integer, primary_key=True)
    patient_id = Column(String, ForeignKey("patient.id"))

    patient = relationship("Patient", back_populates="medical_static")

    diabetes_type = Column(Integer)
    diabetes_duration_years = Column(Float)
    fasting_glucose = Column(Float)
    postprandial_glucose = Column(Float)
    fasting_c_peptide = Column(Float)
    postprandial_c_peptide = Column(Float)
    fasting_insulin = Column(Float)
    postprandial_insulin = Column(Float)
    hba1c = Column(Float)
    glycated_albumin = Column(Float)
    total_cholesterol = Column(Float)
    triglyceride = Column(Float)
    hdl = Column(Float)
    ldl = Column(Float)
    creatinine = Column(Float)
    egfr = Column(Float)
    uric_acid = Column(Float)
    bun = Column(Float)