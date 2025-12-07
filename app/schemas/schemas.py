from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import Optional, Any, List
from datetime import datetime

class SiteCreate(BaseModel):
    name: str
    latitude: float
    longitude: float
    description: Optional[str] = None

class SiteOut(SiteCreate):
    id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class ObservationCreate(BaseModel):
    site_id: int
    timestamp: Optional[datetime] = None
    cloud_cover_pct: Optional[float] = Field(None, ge=0, le=100)
    irradiance_wm2: Optional[float] = None
    energy_output_kw: Optional[float] = None
    meta: Optional[dict] = None

class ObservationOut(ObservationCreate):
    id: int
    model_config = ConfigDict(from_attributes=True)


class UserRegister(BaseModel):
    company_name: str | None = None
    account_type: str | None = None
    email: EmailStr
    password: str = Field(min_length=8)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    company_name: str | None = None
    account_type: str | None = None
    email: EmailStr
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ForgotPassword(BaseModel):
    email: EmailStr


class MessageResponse(BaseModel):
    message: str