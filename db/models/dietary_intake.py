from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from db.base import Base


class DietaryIntake(Base):
    __tablename__ = "dietary_intake"

    id = Column(Integer, primary_key=True, nullable=False)
    patient_id = Column(String, ForeignKey("patient.id"), nullable=False)

    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    hour = Column(Integer)
    minute = Column(Integer)

    patient = relationship("Patient", back_populates="dietary_intake")