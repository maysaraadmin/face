#!/usr/bin/env python3
"""
Test script to verify face embedding extraction and database saving
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')

import numpy as np
from deepface import DeepFace
import contextlib

def test_embedding_extraction():
    """Test face embedding extraction"""
    print("Testing face embedding extraction...")

    try:
        from database import FaceDatabase

        # Initialize database
        db = FaceDatabase()

        # Check current database state
        stats = db.get_database_stats()
        print("📊 Current database stats:")
        print(f"   Total analyses: {stats['total_analyses']}")
        print(f"   Total embeddings: {stats['total_embeddings']}")

        if stats['total_analyses'] == 0:
            print("❌ No analyses in database. Need to analyze some images first.")
            return False

        # Get existing analyses
        analyses = db.get_all_analyses(5)
        print(f"📋 Found {len(analyses)} recent analyses")

        # Check if any have valid image paths
        valid_images = []
        for analysis in analyses:
            image_path = analysis.get('image_path', '')
            if image_path and os.path.exists(image_path):
                valid_images.append(image_path)

        if not valid_images:
            print("❌ No valid image files found for existing analyses")
            print("This suggests the images were moved or deleted after analysis")
            return False

        # Use the first valid image for testing
        test_image = valid_images[0]
        print(f"🖼️ Using existing image: {test_image}")

        # Test DeepFace.represent function
        with contextlib.redirect_stderr(open(os.devnull, 'w')):
            print("🔄 Extracting embedding...")
            embedding_result = DeepFace.represent(test_image, enforce_detection=False, model_name="Facenet")

            if embedding_result and len(embedding_result) > 0:
                embedding_data = embedding_result[0]['embedding']
                facial_area = embedding_result[0]['facial_area']

                print(f"✅ Successfully extracted embedding!")
                print(f"   Embedding dimensions: {len(embedding_data)}")
                print(f"   Embedding type: {type(embedding_data)}")
                print(f"   First 5 values: {embedding_data[:5]}")
                print(f"   Facial area: {facial_area}")

                # Test numpy array conversion
                embedding_array = np.array(embedding_data)
                print(f"   NumPy array shape: {embedding_array.shape}")
                print(f"   NumPy array dtype: {embedding_array.dtype}")

                return True
            else:
                print("❌ No face detected in the image")
                return False

    except Exception as e:
        print(f"❌ Error extracting embedding: {e}")
        return False

def test_database_save():
    """Test saving embedding to database"""
    print("\nTesting database embedding save...")

    try:
        from database import FaceDatabase

        # Initialize database
        db = FaceDatabase()

        # Check current database state
        stats = db.get_database_stats()
        print("📊 Current database stats:")
        print(f"   Total analyses: {stats['total_analyses']}")
        print(f"   Total embeddings: {stats['total_embeddings']}")

        if stats['total_analyses'] == 0:
            print("❌ No analyses in database. Need to analyze some images first.")
            return False

        # Get existing analyses
        analyses = db.get_all_analyses(5)
        print(f"📋 Found {len(analyses)} recent analyses")

        # Check if any have valid image paths
        valid_images = []
        for analysis in analyses:
            image_path = analysis.get('image_path', '')
            if image_path and os.path.exists(image_path):
                valid_images.append(image_path)

        if not valid_images:
            print("❌ No valid image files found for existing analyses")
            print("This suggests the images were moved or deleted after analysis")
            return False

        # Use the first valid image for testing
        test_image = valid_images[0]
        print(f"🖼️ Using existing image: {test_image}")

        # Extract embedding
        with contextlib.redirect_stderr(open(os.devnull, 'w')):
            embedding_result = DeepFace.represent(test_image, enforce_detection=False, model_name="Facenet")

            if embedding_result and len(embedding_result) > 0:
                embedding_data = embedding_result[0]['embedding']
                facial_area = embedding_result[0]['facial_area']

                # Save analysis
                print("💾 Saving analysis to database...")
                analysis_id = db.save_analysis(
                    user_id=None,  # Anonymous user
                    image_path=test_image,
                    analysis_type='analyze',
                    result_data={'test': 'embedding_save_test'},
                    processing_time=1.0,
                    model_used='Facenet',
                    detector_used='opencv'
                )

                print(f"✅ Saved analysis with ID: {analysis_id}")

                # Save embedding
                print("🧠 Saving embedding to database...")
                embedding_id = db.save_embedding(
                    analysis_id=analysis_id,
                    embedding_data=np.array(embedding_data),
                    face_location=facial_area
                )

                print(f"✅ Saved embedding with ID: {embedding_id}")

                # Check database stats after save
                stats_after = db.get_database_stats()
                print("📊 Database stats after save:")
                print(f"   Total analyses: {stats_after['total_analyses']}")
                print(f"   Total embeddings: {stats_after['total_embeddings']}")

                if stats_after['total_embeddings'] > stats['total_embeddings']:
                    print("🎉 SUCCESS: Embedding saved successfully!")
                    return True
                else:
                    print("❌ FAILED: Embedding not saved to database")
                    return False

            else:
                print("❌ No face detected in test image")
                return False

    except Exception as e:
        print(f"❌ Error testing database save: {e}")
        return False

if __name__ == '__main__':
    print("🧪 Starting embedding and database tests...\n")

    success1 = test_embedding_extraction()
    success2 = test_database_save()

    if success1 and success2:
        print("\n🎉 All tests passed! Face search should work correctly.")
    else:
        print("\n❌ Some tests failed. Check the issues above.")
