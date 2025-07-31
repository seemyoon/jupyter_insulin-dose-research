from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from db.base import Base


class AdditionalDrug(Base):
    __tablename__ = "additional_drug"

    id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String, nullable=False)

    patient = relationship("PatientAdditionalDrug", back_populates="drug")
