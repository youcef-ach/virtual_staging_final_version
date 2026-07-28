import sys
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup engine and session
engine = create_engine("sqlite:///./virtual_staging.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

try:
    # We will just insert raw SQL for simplicity to avoid importing all models
    org_id = str(uuid4())
    prop_id = str(uuid4())
    scan_id = str(uuid4())
    
    # 1. Create organization
    db.execute(
        "INSERT INTO organizations (id, name, created_at) VALUES (:id, :name, CURRENT_TIMESTAMP)",
        {"id": org_id, "name": "Demo Organization"}
    )
    
    # 2. Create property
    db.execute(
        "INSERT INTO properties (id, organization_id, name, address, created_at) VALUES (:id, :org_id, :name, :address, CURRENT_TIMESTAMP)",
        {"id": prop_id, "org_id": org_id, "name": "Demo Property", "address": "123 Demo St"}
    )
    
    # 3. Create scan version
    db.execute(
        "INSERT INTO scan_versions (id, property_id, version, status, is_authoritative, raw_assets, completeness, created_at) VALUES (:id, :prop_id, :version, :status, :is_authoritative, :raw_assets, :completeness, CURRENT_TIMESTAMP)",
        {
            "id": scan_id,
            "prop_id": prop_id,
            "version": 1,
            "status": "ingested",
            "is_authoritative": 1,
            "raw_assets": "{}",
            "completeness": '{"complete": true, "has_e57": true, "has_matterpak": true, "has_cubemaps": true}'
        }
    )
    
    db.commit()
    print("Database seeded successfully with Demo Property and Scan Version.")
except Exception as e:
    db.rollback()
    print(f"Error seeding database: {e}")
finally:
    db.close()
