from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text, UniqueConstraint, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.database import Base

class Site(Base):
    __tablename__ = "sites"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    observations = relationship("Observation", back_populates="site", cascade="all, delete-orphan")

class Observation(Base):
    __tablename__ = "observations"
    id = Column(Integer, primary_key=True, index=True)
    site_id = Column(Integer, ForeignKey("sites.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cloud_cover_pct = Column(Float, nullable=True)
    irradiance_wm2 = Column(Float, nullable=True)
    energy_output_kw = Column(Float, nullable=True)
    meta = Column(JSON, nullable=True)

    site = relationship("Site", back_populates="observations")


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
    )

    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), nullable=True)
    account_type = Column(String(50), nullable=True)
    email = Column(String(255), nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    email = Column(String(255), nullable=False)
    success = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)