from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from db.base import Base


class DatasetPartition(Base):
    __tablename__ = 'dataset_partition'

    id = Column(Integer, primary_key=True, nullable=False)

    patients = relationship("Patient", back_populates="dataset_partition")

    name= Column(String, nullable=False, unique=True)