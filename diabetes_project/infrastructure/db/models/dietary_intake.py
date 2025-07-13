from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from diabetes_project.infrastructure.db.base import Base


class DietaryIntake(Base):
    __tablename__ = "dietary_intake"

    id = Column(Integer, primary_key=True)
    patient_id = Column(String, ForeignKey("patient.id"))

    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    hour = Column(Integer)
    minute = Column(Integer)

    intake = Column(Integer)

    patient = relationship("Patient", back_populates="dietary_intake")

    name = Column(String)
