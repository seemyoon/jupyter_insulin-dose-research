from sqlalchemy import Column, Integer
from sqlalchemy.orm import relationship

from diabetes_project.infrastructure.db.base import Base


class DatasetPartition(Base):
    __tablename__ = 'dataset_partition'

    id = Column(Integer, primary_key=True)

    patients = relationship("Patient", back_populates="dataset_partition")