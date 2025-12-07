from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.deps import get_db, get_current_user
from app.core.security import create_access_token
from app.schemas.schemas import UserRegister, UserOut, UserLogin, TokenResponse, ForgotPassword, MessageResponse
from app.services import crud
from app.models.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register_user(payload: UserRegister, db: Session = Depends(get_db)):
    """Register a new user. Email must be a Gmail address."""
    # Validate Gmail address
    if not payload.email.endswith("@gmail.com"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only Gmail addresses are allowed for registration"
        )
    
    try:
        user = crud.create_user(db, payload)
        return user
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    """
    Login with registered email and password. 
    CRITICAL: Only allows registered users with correct credentials.
    Rejects unregistered emails and wrong passwords with 401.
    
    Returns:
    - 200 OK: Login successful, returns TokenResponse with access_token and user
    - 400 BAD REQUEST: Missing email or password
    - 401 UNAUTHORIZED: Invalid email or password
    - 500 INTERNAL SERVER ERROR: Server error
    """
    from app.services.crud import get_user_by_email, verify_password
    import logging
    
    # Log incoming request
    logging.info(f"LOGIN REQUEST RECEIVED for email: {payload.email if payload.email else 'None'}")
    print(f"[BACKEND] Login request received: email={payload.email if payload.email else 'None'}")
    
    # CRITICAL: Validate input - ensure email and password are provided and not empty
    if not payload.email or not payload.email.strip():
        logging.warning("Login attempt with empty email")
        print("[BACKEND] Response: 400 BAD REQUEST - Email is required")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is required"
        )
    
    if not payload.password or not payload.password.strip():
        logging.warning("Login attempt with empty password")
        print("[BACKEND] Response: 400 BAD REQUEST - Password is required")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password is required"
        )
    
    # CRITICAL: Check if user exists first - reject immediately if not found
    email_clean = payload.email.strip().lower()  # Normalize email
    password_clean = payload.password.strip()
    
    print(f"[BACKEND] Checking if user exists: {email_clean}")
    print(f"[BACKEND] Password provided: {'Yes' if password_clean else 'No'}")
    
    # CRITICAL: Query database directly to ensure we get accurate results
    user = get_user_by_email(db, email_clean)
    
    # CRITICAL: Explicit check - user MUST exist, MUST not be None
    if user is None or not hasattr(user, 'id') or not hasattr(user, 'password_hash'):
        # User doesn't exist - reject immediately with 401
        logging.warning(f"Login attempt with non-existent email: {email_clean}")
        print(f"[BACKEND] User NOT FOUND in database for email: {email_clean}")
        print(f"[BACKEND] Response: 401 UNAUTHORIZED - Email not registered")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email not registered. Please register first or check your email address.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # CRITICAL: Verify user object is valid
    if not user.id or not user.password_hash:
        print(f"[BACKEND] Invalid user object - missing id or password_hash")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user account. Please contact support.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    print(f"[BACKEND] User found in database: {email_clean} (ID: {user.id})")
    print(f"[BACKEND] User has password_hash: {'Yes' if user.password_hash else 'No'}")
    
    # CRITICAL: Verify password directly - DO NOT proceed if password is wrong
    print(f"[BACKEND] Verifying password against hash...")
    print(f"[BACKEND] Password hash exists: {bool(user.password_hash)}")
    print(f"[BACKEND] Password hash length: {len(user.password_hash) if user.password_hash else 0}")
    
    password_valid = verify_password(password_clean, user.password_hash)
    
    print(f"[BACKEND] Password verification result: {password_valid}")
    print(f"[BACKEND] Password valid type: {type(password_valid)}")
    print(f"[BACKEND] Password valid is True: {password_valid is True}")
    print(f"[BACKEND] Password valid is False: {password_valid is False}")
    
    # CRITICAL: If password is wrong, reject immediately
    # Use explicit True check - do not rely on truthiness
    if password_valid is not True:
        # Password is wrong - explicitly reject - do not create token, do not proceed
        logging.warning(f"Login failed for user: {email_clean} - invalid password")
        print(f"[BACKEND] Password verification FAILED")
        print(f"[BACKEND] Response: 401 UNAUTHORIZED - Incorrect password")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password. Please check your password and try again.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    print(f"[BACKEND] Password verified successfully")
    
    # CRITICAL: Final validation before creating token
    # Only reach here if BOTH checks passed:
    # 1. User exists in database (checked above) ✓
    # 2. Password is correct (checked above) ✓
    
    # CRITICAL: Double-check user is valid before creating token
    if not user or not user.id or not user.password_hash:
        print(f"[BACKEND] CRITICAL: User object invalid before token creation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication"
        )
    
    # CRITICAL: Only create token if BOTH email exists AND password is correct
    # User authenticated successfully - create access token
    logging.info(f"Login successful for user: {email_clean} (ID: {user.id})")
    print(f"[BACKEND] Authentication successful for user: {email_clean}")
    print(f"[BACKEND] User ID: {user.id}, Email: {user.email}")
    
    access_token = create_access_token(data={"sub": user.id})
    print(f"[BACKEND] Access token created for user ID: {user.id}")
    
    response_data = TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=user
    )
    
    print(f"[BACKEND] Response: 200 OK - Returning TokenResponse")
    print(f"[BACKEND] Response data:")
    print(f"   - token_type: {response_data.token_type}")
    print(f"   - user_id: {response_data.user.id}")
    print(f"   - user_email: {response_data.user.email}")
    print(f"   - access_token: {response_data.access_token[:20]}...")
    
    # CRITICAL: Return response only after successful authentication
    # This should ONLY happen if:
    # 1. Email exists in database ✓
    # 2. Password matches hash ✓
    return response_data


@router.get("/me", response_model=UserOut)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current authenticated user's profile."""
    return current_user


@router.get("/profile/{user_id}", response_model=UserOut)
def get_user_profile(user_id: int, current_user: User = Depends(get_current_user)):
    """Get user profile by ID. Users can only view their own profile."""
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this profile"
        )
    return current_user


@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(payload: ForgotPassword, db: Session = Depends(get_db)):
    """Request password reset. Sends reset instructions to the user's email if the email is registered."""
    from app.services.crud import get_user_by_email
    import logging
    
    email_clean = payload.email.strip().lower()
    
    # Check if user exists
    user = get_user_by_email(db, email_clean)
    
    # Always return success message for security (don't reveal if email exists)
    # In production, you would send an email with reset link here
    if user:
        logging.info(f"Password reset requested for user: {email_clean}")
        print(f"[BACKEND] Password reset requested for registered email: {email_clean}")
        # TODO: Send password reset email with token/link
        # For now, just log the request
    else:
        logging.warning(f"Password reset requested for non-existent email: {email_clean}")
        print(f"[BACKEND] Password reset requested for non-existent email: {email_clean}")
        # Still return success to prevent email enumeration
    
    return MessageResponse(
        message="If an account with that email exists, password reset instructions have been sent."
    )

