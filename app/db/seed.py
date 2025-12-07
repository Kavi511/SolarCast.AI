from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.models.models import User, LoginAttempt


def seed_dummy_users(db: Session) -> int:
    """Insert demo users if no users exist. Returns created count."""
    existing_count = db.query(User).count()
    if existing_count > 0:
        return 0

    # No demo users - users must register through the API
    db.commit()
    return 0


def seed_login_attempts(db: Session) -> int:
    """Insert demo login attempts. Returns created count."""
    # Get existing users
    users = db.query(User).all()
    if not users:
        print("No users found. Please seed users first.")
        return 0

    # Clear existing login attempts (optional - comment out if you want to keep them)
    db.query(LoginAttempt).delete()
    db.commit()

    now = datetime.utcnow()
    attempts = []

    # Successful login attempts for demo users
    for i, user in enumerate(users[:3]):  # First 3 users
        attempts.append(LoginAttempt(
            user_id=user.id,
            email=user.email,
            success=True,
            created_at=now - timedelta(hours=i*2)
        ))

    # Failed login attempts (wrong password)
    attempts.append(LoginAttempt(
        user_id=users[0].id,
        email=users[0].email,
        success=False,
        created_at=now - timedelta(hours=1)
    ))

    # Failed login attempts (non-existent email)
    attempts.append(LoginAttempt(
        user_id=None,
        email="hacker@example.com",
        success=False,
        created_at=now - timedelta(minutes=30)
    ))
    attempts.append(LoginAttempt(
        user_id=None,
        email="test@fake.com",
        success=False,
        created_at=now - timedelta(minutes=15)
    ))

    # More successful attempts
    attempts.append(LoginAttempt(
        user_id=users[1].id,
        email=users[1].email,
        success=True,
        created_at=now - timedelta(minutes=10)
    ))
    attempts.append(LoginAttempt(
        user_id=users[2].id,
        email=users[2].email,
        success=True,
        created_at=now - timedelta(minutes=5)
    ))

    for attempt in attempts:
        db.add(attempt)

    db.commit()
    return len(attempts)


