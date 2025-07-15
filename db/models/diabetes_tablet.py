from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from db.base import Base


class DiabetesTablets(Base):
    __tablename__ = 'diabetes_tablet'

    id = Column(Integer, primary_key=True, nullable=False)

    taking_tablets = relationship("TakingDiabetesTablet", back_populates="diabetes_tablet")

    name = Column(String)
