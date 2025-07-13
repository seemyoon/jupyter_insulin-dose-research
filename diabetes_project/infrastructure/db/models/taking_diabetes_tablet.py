from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship

from diabetes_project.infrastructure.db.base import Base


class TakingDiabetesTablet(Base):
    __tablename__ = 'taking_diabetes_tablet'

    id = Column(Integer, primary_key=True)
    patient_id = Column(String, ForeignKey("patient.id"))
    diabetes_tablet_id = Column(Integer, ForeignKey("diabetes_tablet.id"))

    patient = relationship("Patient", back_populates="tablets")
    diabetes_tablet = relationship("DiabetesTablets", back_populates="taking_tablets")

    dose = Column(Float)
