from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.models import Site, Observation, User, LoginAttempt
from app.schemas.schemas import SiteCreate, ObservationCreate, UserRegister
from datetime import datetime
import bcrypt

# We use bcrypt directly here so we avoid passlib compatibility surprises.
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    # Create a salt, then hash the password with it.
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash using bcrypt.
    
    Supports both direct bcrypt hashes and passlib-formatted hashes.
    Passlib bcrypt hashes are compatible with bcrypt.checkpw.
    
    This returns True only for a real match; everything else is False.
    """
    # If either value is missing, fail fast.
    if not plain_password or not hashed_password:
        import logging
        logging.warning("Password verification: Missing password or hash")
        return False
    
    # Reject whitespace-only passwords as invalid input.
    if not plain_password.strip():
        import logging
        logging.warning("Password verification: Empty password")
        return False
    
    try:
        # bcrypt.checkpw works for both native bcrypt hashes and passlib bcrypt hashes.
        password_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8')
        
        # Run the actual hash comparison.
        result = bcrypt.checkpw(password_bytes, hash_bytes)
        
        # Keep a debug trail so auth issues are easier to diagnose.
        import logging
        logging.debug(f"bcrypt.checkpw result: {result}, type: {type(result)}")
        
        # Return a strict boolean so callers never get None by accident.
        if result is True:
            logging.info("Password verification: Password matches hash - RETURNING TRUE")
            return True
        else:
            # Any non-True result means the password check failed.
            logging.warning(f"Password verification: Password does not match hash - result={result}, type={type(result)}")
            return False
    except (ValueError, TypeError, AttributeError) as e:
        # Handle common verification errors and treat them as failed auth.
        import logging
        logging.error("Password verification error: %s", str(e))
        return False
    except Exception as e:
        # Catch unexpected issues and still fail safely.
        import logging
        logging.error("Unexpected password verification error: %s", str(e))
        return False

def create_site(db: Session, data: SiteCreate) -> Site:
    site = Site(**data.model_dump())
    db.add(site)
    db.commit()
    db.refresh(site)
    return site

def get_site(db: Session, site_id: int) -> Optional[Site]:
    return db.query(Site).filter(Site.id==site_id).first()

def list_sites(db: Session, skip: int=0, limit: int=100) -> List[Site]:
    return db.query(Site).offset(skip).limit(limit).all()

def delete_site(db: Session, site_id: int) -> bool:
    site = get_site(db, site_id)
    if not site: return False
    db.delete(site)
    db.commit()
    return True

def add_observation(db: Session, data: ObservationCreate) -> Observation:
    obs = Observation(**data.model_dump())
    db.add(obs)
    db.commit()
    db.refresh(obs)
    return obs

def list_observations(db: Session, site_id: int, skip: int=0, limit: int=200):
    q = db.query(Observation).filter(Observation.site_id==site_id).order_by(Observation.timestamp.desc())
    return q.offset(skip).limit(limit).all()


def get_user_by_email(db: Session, email: str) -> User | None:
    """Get user by email. Returns None if user doesn't exist."""
    # Normalize once so email lookup is reliably case-insensitive.
    if not email:
        import logging
        logging.warning("get_user_by_email: Empty email provided")
        return None
    
    email_normalized = email.strip().lower()
    if not email_normalized:
        import logging
        logging.warning("get_user_by_email: Email is empty after normalization")
        return None
    
    # Query with a lowercased comparison at the database level.
    from sqlalchemy import func
    user = db.query(User).filter(func.lower(User.email) == email_normalized).first()
    
    # Double-check the email match before returning.
    if user:
        # Guard against unexpected mismatches.
        if user.email.lower() != email_normalized:
            # If the values do not actually match, treat as not found.
            import logging
            logging.warning(f"get_user_by_email: Email mismatch - query: {email_normalized}, found: {user.email}")
            return None
        
        # Make sure we have the minimum fields needed for auth flows.
        if not hasattr(user, 'id') or not hasattr(user, 'password_hash'):
            import logging
            logging.error(f"get_user_by_email: User found but missing required fields")
            return None
        
        import logging
        logging.info(f"get_user_by_email: User found - email: {user.email}, id: {user.id}")
        return user
    
    # No matching user for this email.
    import logging
    logging.info(f"get_user_by_email: No user found for email: {email_normalized}")
    return None


def create_user(db: Session, payload: UserRegister) -> User:
    existing = get_user_by_email(db, payload.email)
    if existing:
        raise ValueError("Email already registered")

    password_hash = hash_password(payload.password)
    user = User(
        company_name=payload.company_name,
        account_type=payload.account_type,
        email=payload.email,
        password_hash=password_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    """Authenticate user and log login attempt."""
    user = get_user_by_email(db, email)
    
    # Record this attempt whether it succeeds or fails.
    attempt = LoginAttempt(
        user_id=user.id if user else None,
        email=email,
        success=False,
        created_at=datetime.utcnow()
    )
    
    # If no user exists, store the failed attempt and stop here.
    if not user:
        db.add(attempt)
        db.commit()
        return None
    
    # Password verification decides whether login is allowed.
    try:
        import logging
        logging.info(f"Authenticating user: {email}")
        password_valid = verify_password(password, user.password_hash)
        
        if password_valid is True:
            # Successful password check: mark attempt and return user.
            logging.info(f"Password verified successfully for user: {email}")
            attempt.success = True
            db.add(attempt)
            db.commit()
            return user
        else:
            # Failed password check: keep login denied.
            logging.warning(f"Password verification FAILED for user: {email}")
            db.add(attempt)
            db.commit()
            return None  # Returning None keeps authentication denied.
    except Exception as e:
        # Any verification error is treated as a failed login.
        import logging
        logging.error(f"Password verification exception for user {email}: {e}")
        db.add(attempt)
        db.commit()
        return None  # Returning None keeps authentication denied.


def get_user_by_id(db: Session, user_id: int) -> User | None:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()