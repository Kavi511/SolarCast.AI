from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.models import Site, Observation, User, LoginAttempt
from app.schemas.schemas import SiteCreate, ObservationCreate, UserRegister
from datetime import datetime
import bcrypt

# Use bcrypt directly instead of passlib to avoid compatibility issues
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    # Generate salt and hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash using bcrypt.
    
    Supports both direct bcrypt hashes and passlib-formatted hashes.
    Passlib bcrypt hashes are compatible with bcrypt.checkpw.
    
    CRITICAL: Returns False for ANY invalid password - never returns True unless password matches.
    """
    # CRITICAL: Return False immediately if inputs are invalid
    if not plain_password or not hashed_password:
        import logging
        logging.warning("Password verification: Missing password or hash")
        return False
    
    # CRITICAL: Ensure password is not empty string
    if not plain_password.strip():
        import logging
        logging.warning("Password verification: Empty password")
        return False
    
    try:
        # bcrypt.checkpw works with both direct bcrypt hashes and passlib bcrypt hashes
        # Both use the same underlying bcrypt format ($2b$12$...)
        password_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8')
        
        # CRITICAL: Use bcrypt.checkpw - this is the ONLY way to verify
        # It returns True ONLY if password matches hash exactly
        result = bcrypt.checkpw(password_bytes, hash_bytes)
        
        # CRITICAL: Log the result for debugging
        import logging
        logging.debug(f"bcrypt.checkpw result: {result}, type: {type(result)}")
        
        # CRITICAL: Ensure we return a boolean - never None
        # Use explicit True check - do not rely on truthiness
        if result is True:
            logging.info("Password verification: Password matches hash - RETURNING TRUE")
            return True
        else:
            # result is False or None - reject
            logging.warning(f"Password verification: Password does not match hash - result={result}, type={type(result)}")
            return False
    except (ValueError, TypeError, AttributeError) as e:
        # Handle specific exceptions that might occur during verification
        import logging
        logging.error("Password verification error: %s", str(e))
        return False
    except Exception as e:
        # Catch any other unexpected errors
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
    # CRITICAL: Normalize email to lowercase for case-insensitive lookup
    if not email:
        import logging
        logging.warning("get_user_by_email: Empty email provided")
        return None
    
    email_normalized = email.strip().lower()
    if not email_normalized:
        import logging
        logging.warning("get_user_by_email: Email is empty after normalization")
        return None
    
    # CRITICAL: Query database with exact match (case-insensitive)
    # Use func.lower() for database-level case-insensitive comparison
    from sqlalchemy import func
    user = db.query(User).filter(func.lower(User.email) == email_normalized).first()
    
    # CRITICAL: Double-check the email matches exactly (case-insensitive)
    if user:
        # Verify the email actually matches (case-insensitive comparison)
        if user.email.lower() != email_normalized:
            # Email doesn't match - return None
            import logging
            logging.warning(f"get_user_by_email: Email mismatch - query: {email_normalized}, found: {user.email}")
            return None
        
        # CRITICAL: Verify user has required fields
        if not hasattr(user, 'id') or not hasattr(user, 'password_hash'):
            import logging
            logging.error(f"get_user_by_email: User found but missing required fields")
            return None
        
        import logging
        logging.info(f"get_user_by_email: User found - email: {user.email}, id: {user.id}")
        return user
    
    # User not found
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
    
    # Log login attempt
    attempt = LoginAttempt(
        user_id=user.id if user else None,
        email=email,
        success=False,
        created_at=datetime.utcnow()
    )
    
    # If user doesn't exist, log attempt and return None
    if not user:
        db.add(attempt)
        db.commit()
        return None
    
    # CRITICAL: Verify password - this is the critical check
    try:
        import logging
        logging.info(f"Authenticating user: {email}")
        password_valid = verify_password(password, user.password_hash)
        
        if password_valid is True:
            # Password is correct
            logging.info(f"Password verified successfully for user: {email}")
            attempt.success = True
            db.add(attempt)
            db.commit()
            return user
        else:
            # CRITICAL: Password verification returned False - reject login
            logging.warning(f"Password verification FAILED for user: {email}")
            db.add(attempt)
            db.commit()
            return None  # CRITICAL: Return None to reject login
    except Exception as e:
        # If verification throws an exception, log it and reject
        import logging
        logging.error(f"Password verification exception for user {email}: {e}")
        db.add(attempt)
        db.commit()
        return None  # CRITICAL: Return None to reject login


def get_user_by_id(db: Session, user_id: int) -> User | None:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()