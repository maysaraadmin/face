#!/usr/bin/env python3
"""
Database Setup and Testing Script for DeepFace GUI
"""

import sys
import os
from database import FaceDatabase
import numpy as np

def test_database():
    """Test database functionality"""
    print("🗄️  Testing Face Database...")

    # Initialize database
    db = FaceDatabase()

    try:
        # Test user creation
        print("👤 Adding test users...")
        user1_id = db.add_user("John Doe", "john@example.com")
        user2_id = db.add_user("Jane Smith", "jane@example.com")
        user3_id = db.add_user("Anonymous User")  # No email

        print(f"✅ Users created/updated: {user1_id}, {user2_id}, {user3_id}")

        # Test database statistics
        print("\n📊 Database Statistics:")
        stats = db.get_database_stats()
        for key, value in stats.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")

        # Test verification history
        print("\n⚖️  Recent Verifications:")
        verifications = db.get_verification_history()
        print(f"   Found {len(verifications)} verification records")

        print("\n✅ Database test completed successfully!")
        print(f"📁 Database file: {db.db_path}")
        print(f"📏 Database size: {stats['database_size_mb']:.2f} MB")

        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

    finally:
        db.close()

def create_sample_data():
    """Create sample data for testing"""
    print("\n📝 Creating sample data...")

    db = FaceDatabase()

    try:
        # Add sample users with unique emails
        import time
        timestamp = int(time.time())

        users = [
            ("Alice Johnson", f"alice_{timestamp}@example.com"),
            ("Bob Wilson", f"bob_{timestamp}@example.com"),
            ("Carol Brown", f"carol_{timestamp}@example.com"),
            ("David Lee", None),  # No email
            ("Emma Davis", f"emma_{timestamp}@example.com")
        ]

        user_ids = []
        for name, email in users:
            user_id = db.add_user(name, email)
            user_ids.append(user_id)
            print(f"   Added user: {name} (ID: {user_id})")

        # Create sample analysis results
        print("\n🔍 Creating sample analysis results...")

        sample_analysis = {
            "age": 28,
            "gender": "Woman",
            "race": {
                "asian": 0.15,
                "black": 0.05,
                "hispanic": 0.10,
                "middle eastern": 0.05,
                "white": 0.65
            },
            "emotion": {
                "angry": 0.02,
                "disgust": 0.01,
                "fear": 0.01,
                "happy": 0.85,
                "neutral": 0.08,
                "sad": 0.02,
                "surprise": 0.01
            },
            "model": "VGG-Face",
            "detector_backend": "opencv",
            "time": 1.23
        }

        # Save sample analyses
        for i, user_id in enumerate(user_ids[:3]):  # First 3 users
            analysis_id = db.save_analysis(
                user_id=user_id,
                image_path=f"sample_image_{i+1}.jpg",
                analysis_type="analyze",
                result_data=sample_analysis,
                processing_time=1.23,
                model_used="VGG-Face",
                detector_used="opencv"
            )
            print(f"   Saved analysis for user {user_id}: ID {analysis_id}")

        print("\n✅ Sample data created successfully!")

        # Show updated statistics
        stats = db.get_database_stats()
        print("\n📊 Updated Statistics:")
        print(f"   Total Users: {stats['total_users']}")
        print(f"   Total Analyses: {stats['total_analyses']}")
        print(f"   Database Size: {stats['database_size_mb']:.2f} MB")

    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

    finally:
        db.close()

    return True

def main():
    """Main setup function"""
    print("🚀 DeepFace Database Setup\n")

    # Test database
    if not test_database():
        print("❌ Database setup failed!")
        return False

    # Ask user if they want sample data
    print("\n❓ Would you like to create sample data for testing? (y/n)")
    try:
        create_samples = input().lower().strip() in ['y', 'yes']
    except:
        create_samples = False

    if create_samples:
        create_sample_data()

    print("\n✅ Setup completed successfully!")
    print("🎯 You can now run: python deepface_gui.py")
    print("🗄️  Database manager: python database_manager.py")

    return True

if __name__ == "__main__":
    main()
