from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from diabetes_project.infrastructure.db.base import Base


class DiabetesTablets(Base):
    __tablename__ = 'diabetes_tablet'

    id = Column(Integer, primary_key=True)

    taking_tablets = relationship("TakingDiabetesTablet", back_populates="diabetes_tablet")

    name = Column(String)
