"""Force seed users - deletes existing users and recreates them."""
import os
import sys
import getpass


# Instructions
print("=" * 60)
print("Force Seed Users Script")
print("=" * 60)
print()

# Set default database environment variables if not set
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_DB", "sola_ai")

# Check if password is set, if not prompt for it
if not os.getenv("POSTGRES_PASSWORD"):
    print("POSTGRES_PASSWORD not found in environment.")
    print("Please enter your PostgreSQL password:")
    password = getpass.getpass("Password: ")
    if password:
        os.environ["POSTGRES_PASSWORD"] = password
        print("✓ Password set")
    else:
        print("ERROR: Password cannot be empty!")
        input("\nPress Enter to exit...")
        sys.exit(1)
else:
    print("✓ Using POSTGRES_PASSWORD from environment")

print(f"\nDatabase Configuration:")
print(f"  Host: {os.getenv('POSTGRES_HOST')}")
print(f"  Port: {os.getenv('POSTGRES_PORT')}")
print(f"  User: {os.getenv('POSTGRES_USER')}")
print(f"  Database: {os.getenv('POSTGRES_DB')}")
print()

from app.db.database import SessionLocal, engine, Base
from app.db.seed import seed_dummy_users
from app.models.models import User
from app.services.crud import verify_password, get_user_by_email

def main():
    print("\nConnecting to database...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    db = SessionLocal()
    try:
        # Delete all existing users
        existing_count = db.query(User).count()
        if existing_count > 0:
            print(f"\nFound {existing_count} existing users. Deleting...")
            db.query(User).delete()
            db.commit()
            print("✓ Deleted existing users")
        
        # Seed new users
        print("\nCreating demo users...")
        count = seed_dummy_users(db)
        print(f"✓ Created {count} demo users")
        
        # Verify and test
        users = db.query(User).all()
        print(f"\n✓ Total users in database: {len(users)}")
        print("\nCreated users:")
        for user in users:
            print(f"  - {user.email} ({user.company_name})")
        
        # Test password
        test_user = get_user_by_email(db, "demo1@gmail.com")
        if test_user:
            is_valid = verify_password("DemoPass123!", test_user.password_hash)
            print(f"\nPassword test for demo1@gmail.com: {'✓ VALID' if is_valid else '✗ INVALID'}")
        
        print("\n" + "=" * 60)
        print("SUCCESS! You can now log in with:")
        print("=" * 60)
        print("Email: demo1@gmail.com")
        print("Password: DemoPass123!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()

