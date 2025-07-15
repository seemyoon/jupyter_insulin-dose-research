from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship

from db.base import Base


class TakingDiabetesTablet(Base):
    __tablename__ = 'taking_diabetes_tablet'

    id = Column(Integer, primary_key=True, nullable=False)
    patient_id = Column(String, ForeignKey("patient.id"), nullable=False)
    diabetes_tablet_id = Column(Integer, ForeignKey("diabetes_tablet.id"), nullable=False)

    patient = relationship("Patient", back_populates="tablets")
    diabetes_tablet = relationship("DiabetesTablets", back_populates="taking_tablets")

    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    hour = Column(Integer)
    minute = Column(Integer)

    dose = Column(Float)
