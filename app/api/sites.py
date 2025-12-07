from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.core.deps import get_db
from app.schemas.schemas import SiteCreate, SiteOut, ObservationCreate, ObservationOut
from app.services import crud

router = APIRouter(prefix="/sites", tags=["Sites"])

@router.post("", response_model=SiteOut)
def create_site(payload: SiteCreate, db: Session = Depends(get_db)):
    return crud.create_site(db, payload)

@router.get("", response_model=List[SiteOut])
def list_sites(db: Session = Depends(get_db)):
    return crud.list_sites(db)

@router.get("/{site_id}", response_model=SiteOut)
def get_site(site_id: int, db: Session = Depends(get_db)):
    site = crud.get_site(db, site_id)
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    return site

@router.delete("/{site_id}")
def delete_site(site_id: int, db: Session = Depends(get_db)):
    ok = crud.delete_site(db, site_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Site not found")
    return {"status": "deleted"}

@router.post("/{site_id}/observations", response_model=ObservationOut)
def add_observation(site_id: int, payload: ObservationCreate, db: Session = Depends(get_db)):
    if payload.site_id != site_id:
        payload.site_id = site_id
    return crud.add_observation(db, payload)

@router.get("/{site_id}/observations", response_model=List[ObservationOut])
def list_observations(site_id: int, db: Session = Depends(get_db)):
    return crud.list_observations(db, site_id)