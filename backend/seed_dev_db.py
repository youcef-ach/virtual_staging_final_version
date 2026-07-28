"""Seed the database with a test property and scan version."""

from sqlalchemy.orm import Session
from app.db import get_engine, Base
from app.models import Organization, User, Property, ScanVersion, UserRole, ScanStatus

def seed_db():
    engine = get_engine()
    # Ensure tables exist
    Base.metadata.create_all(engine)
    
    with Session(engine) as db:
        # Create organization
        org = db.query(Organization).first()
        if not org:
            org = Organization(name="Test Org")
            db.add(org)
            db.commit()
            db.refresh(org)
            print(f"Created organization: {org.id}")
            
        # Create user
        user = db.query(User).first()
        if not user:
            user = User(
                organization_id=org.id,
                email="admin@test.com",
                password_hash="fake",
                role=UserRole.ADMIN
            )
            db.add(user)
            db.commit()
            print(f"Created user: {user.email}")
            
        # Check if property already exists
        prop = db.query(Property).first()
        if not prop:
            prop = Property(
                organization_id=org.id,
                name="Demo House",
                address="123 Test Ave"
            )
            db.add(prop)
            db.commit()
            db.refresh(prop)
            print(f"Created property: {prop.id}")
        else:
            print(f"Using existing property: {prop.id}")
            
        # Check if scan exists
        scan = db.query(ScanVersion).filter_by(property_id=prop.id).first()
        if not scan:
            scan = ScanVersion(
                property_id=prop.id,
                version=1,
                status=ScanStatus.INGESTED,
                is_authoritative=True,
                raw_assets={}
            )
            db.add(scan)
            db.commit()
            db.refresh(scan)
            print(f"Created scan version: {scan.id}")
        else:
            print(f"Using existing scan version: {scan.id}")

if __name__ == "__main__":
    seed_db()
