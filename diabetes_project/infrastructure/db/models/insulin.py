from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from diabetes_project.infrastructure.db.base import Base


class Insulin(Base):
    __tablename__ = 'insulin'

    id = Column(Integer, primary_key=True)

    taking_insulins = relationship("TakingInsulin", back_populates="insulin")

    name = Column(String)