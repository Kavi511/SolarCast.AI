"""Script to check and seed users in the database."""
import os
import sys

# Set environment variables if not set
if not os.getenv("POSTGRES_PASSWORD"):
    print("WARNING: POSTGRES_PASSWORD not set. Please set it or run from start.ps1")
    print("Setting default values...")
    os.environ.setdefault("POSTGRES_HOST", "localhost")
    os.environ.setdefault("POSTGRES_PORT", "5432")
    os.environ.setdefault("POSTGRES_USER", "postgres")
    os.environ.setdefault("POSTGRES_DB", "sola_ai")
    print("NOTE: You may need to set POSTGRES_PASSWORD manually")

from app.db.database import SessionLocal, engine, Base
from app.db.seed import seed_dummy_users
from app.models.models import User
from app.services.crud import verify_password, get_user_by_email

def main():
    print("=" * 50)
    print("Checking Database Users")
    print("=" * 50)
    
    try:
        # Test database connection
        print("\nTesting database connection...")
        Base.metadata.create_all(bind=engine)
        print("✓ Database connection successful")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running")
        print("  2. POSTGRES_PASSWORD environment variable is set")
        print("  3. Database 'sola_ai' exists")
        sys.exit(1)
    
    db = SessionLocal()
    try:
        # Check existing users
        users = db.query(User).all()
        print(f"\nFound {len(users)} users in database:")
        
        if users:
            for user in users:
                print(f"  - ID: {user.id}, Email: {user.email}, Company: {user.company_name}")
        else:
            print("  No users found!")
        
        # Seed users if none exist
        if len(users) == 0:
            print("\nSeeding demo users...")
            count = seed_dummy_users(db)
            print(f"✓ Created {count} demo users")
            
            # Verify the users were created
            users = db.query(User).all()
            print(f"\nNow found {len(users)} users:")
            for user in users:
                print(f"  - ID: {user.id}, Email: {user.email}, Company: {user.company_name}")
        else:
            print("\nUsers already exist. Testing password verification...")
            test_user = get_user_by_email(db, "demo1@gmail.com")
            if test_user:
                test_password = "DemoPass123!"
                is_valid = verify_password(test_password, test_user.password_hash)
                print(f"\nTesting password for demo1@gmail.com:")
                print(f"  Password: {test_password}")
                print(f"  Verification result: {'✓ VALID' if is_valid else '✗ INVALID'}")
            else:
                print("\n⚠ demo1@gmail.com not found!")
        
        print("\n" + "=" * 50)
        print("Test Credentials:")
        print("=" * 50)
        print("Email: demo1@gmail.com")
        print("Password: DemoPass123!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    main()

