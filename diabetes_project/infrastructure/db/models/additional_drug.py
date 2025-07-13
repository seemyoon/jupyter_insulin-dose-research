from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from diabetes_project.infrastructure.db.base import Base


class AdditionalDrugs(Base):
    __tablename__ = "additional_drug"

    id = Column(Integer, primary_key=True)
    patient_id = Column(String, ForeignKey("patient.id"))

    patient = relationship("Patient", back_populates="additional_drugs")

    name = Column(String)
