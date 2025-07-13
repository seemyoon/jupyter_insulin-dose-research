from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship

from diabetes_project.infrastructure.db.base import Base


class Measurement(Base):
    __tablename__ = "measurement"

    id = Column(Integer, primary_key=True)
    patient_id = Column(String, ForeignKey("patient.id"))

    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    hour = Column(Integer)
    minute = Column(Integer)

    cgm = Column(Float)
    cbg = Column(Float)
    blood_ketone = Column(Float)

    patients = relationship("Patient", back_populates="measurements")

