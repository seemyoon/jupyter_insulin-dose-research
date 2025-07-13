from sqlalchemy import Column, String, Integer, ForeignKey, Float
from sqlalchemy.orm import relationship

from diabetes_project.infrastructure.db.base import Base


class Patient(Base):
    __tablename__ = 'patient'

    id = Column(String, primary_key=True)
    dataset_partition_id = Column(Integer, ForeignKey("dataset_partition.id"))

    dataset_partition = relationship("DatasetPartition", back_populates="patients")

    insulins = relationship("TakingInsulin", back_populates="patient")
    tablets = relationship("TakingDiabetesTablet", back_populates="patient")
    additional_drugs = relationship("AdditionalDrugs", back_populates="patient")
    comorbidities = relationship("Comorbidities", back_populates="patient")
    medical_static = relationship("PatientMedicalStatic", back_populates="patient", uselist=False)

    measurements = relationship("Measurement", back_populates="patient")

    gender = Column(Integer)
    age = Column(Integer)
    height = Column(Float)
    weight = Column(Float)
    smoking_history = Column(Integer)
    alcohol_drinking_history = Column(Integer)
