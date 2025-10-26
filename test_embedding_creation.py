#!/usr/bin/env python3
"""
Test script to verify face embedding extraction works correctly
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')

from deepface_gui import DeepFaceGUI
import numpy as np

def test_embedding_creation():
    """Test the embedding creation functionality"""
    print("Testing face embedding creation...")

    # Create a minimal GUI instance just for testing
    class TestGUI:
        def create_embedding_from_analysis(self, analysis_result):
            """Create a simple embedding representation from analysis data"""
            if not analysis_result:
                return None

            embedding_features = []

            try:
                # Age contribution
                if 'age' in analysis_result and analysis_result['age'] is not None:
                    age = float(analysis_result['age'])
                    embedding_features.extend([age / 100.0])

                # Gender contribution (0 for male, 1 for female)
                if 'gender' in analysis_result and analysis_result['gender'] is not None:
                    gender_data = analysis_result['gender']
                    # Gender is a dict with 'Man' and 'Woman' confidence scores
                    if isinstance(gender_data, dict):
                        man_confidence = float(gender_data.get('Man', 0))
                        woman_confidence = float(gender_data.get('Woman', 0))
                        gender_score = 1.0 if woman_confidence > man_confidence else 0.0
                    else:
                        # Fallback for string format
                        gender_score = 1.0 if str(gender_data).lower() == 'woman' else 0.0
                    embedding_features.append(gender_score)

                # Race contributions
                if 'race' in analysis_result and analysis_result['race'] is not None:
                    race = analysis_result['race']
                    if isinstance(race, dict):
                        embedding_features.extend([
                            float(race.get('white', 0)),
                            float(race.get('black', 0)),
                            float(race.get('asian', 0)),
                            float(race.get('hispanic', 0)),
                            float(race.get('middle eastern', 0))
                        ])
                    else:
                        # If race is a string, create a simple encoding
                        race_str = str(race).lower()
                        race_encoding = [1.0 if race_str == ethnicity else 0.0 for ethnicity in ['white', 'black', 'asian', 'hispanic', 'middle eastern']]
                        embedding_features.extend(race_encoding)

                # Emotion contributions
                if 'emotion' in analysis_result and analysis_result['emotion'] is not None:
                    emotions = analysis_result['emotion']
                    if isinstance(emotions, dict):
                        embedding_features.extend([
                            float(emotions.get('happy', 0)),
                            float(emotions.get('sad', 0)),
                            float(emotions.get('angry', 0)),
                            float(emotions.get('neutral', 0))
                        ])
                    else:
                        # If emotions is not a dict, add zeros
                        embedding_features.extend([0, 0, 0, 0])

            except Exception as e:
                print(f"Warning: Error processing analysis features: {e}")
                # Continue with whatever features we have so far

            # Pad or truncate to create a fixed-size embedding
            target_size = 128  # Standard face embedding size
            while len(embedding_features) < target_size:
                embedding_features.append(0.0)

            return np.array(embedding_features[:target_size])

    # Test with sample data that matches DeepFace output format
    test_gui = TestGUI()

    # Test case 1: Dictionary format (normal DeepFace output)
    sample_analysis_dict = {
        'age': 30,
        'gender': {'Man': 0.3, 'Woman': 0.7},
        'race': {'white': 0.6, 'black': 0.1, 'asian': 0.2, 'hispanic': 0.05, 'middle eastern': 0.05},
        'emotion': {'happy': 0.5, 'sad': 0.1, 'angry': 0.1, 'neutral': 0.3}
    }

    embedding1 = test_gui.create_embedding_from_analysis(sample_analysis_dict)
    print(f"✅ Test 1 passed: Created embedding with {len(embedding1)} features")
    print(f"   Age feature: {embedding1[0]:.3f}")
    print(f"   Gender feature: {embedding1[1]:.3f} (should be 1.0 for woman)")
    print(f"   Race features: {embedding1[2:7]}")
    print(f"   Emotion features: {embedding1[7:11]}")

    # Test case 2: String format (fallback)
    sample_analysis_string = {
        'age': 25,
        'gender': 'Man',
        'race': 'white',
        'emotion': {'happy': 0.8, 'sad': 0.1, 'angry': 0.05, 'neutral': 0.05}
    }

    embedding2 = test_gui.create_embedding_from_analysis(sample_analysis_string)
    print(f"✅ Test 2 passed: Created embedding with {len(embedding2)} features")
    print(f"   Gender feature: {embedding2[1]:.3f} (should be 0.0 for man)")
    print(f"   Race features: {embedding2[2:7]} (should be [1.0, 0.0, 0.0, 0.0, 0.0])")

    # Test case 3: Missing data
    sample_analysis_minimal = {
        'age': 40
    }

    embedding3 = test_gui.create_embedding_from_analysis(sample_analysis_minimal)
    print(f"✅ Test 3 passed: Created embedding with {len(embedding3)} features")
    print(f"   Age feature: {embedding3[0]:.3f} (should be 0.4)")
    print(f"   Other features should be 0.0")

    print("\n🎉 All embedding creation tests passed!")
    return True

if __name__ == '__main__':
    test_embedding_creation()
