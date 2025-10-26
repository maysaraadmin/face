import sqlite3
import json
import os
from datetime import datetime
import numpy as np
from typing import Optional, List, Dict, Any
from PIL import Image

class FaceDatabase:
    """Database manager for storing face analysis results and user data"""

    def __init__(self, db_path: str = "face_database.db"):
        self.db_path = db_path
        self.init_database()

    def numpy_to_python_types(self, obj):
        """Convert NumPy types to JSON-serializable Python types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self.numpy_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.numpy_to_python_types(item) for item in obj]
        else:
            return obj

    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if users table exists and its schema
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='users'")
            users_table = cursor.fetchone()

            if users_table and 'email TEXT UNIQUE' in users_table[0]:
                # Old schema - need to migrate
                print("Migrating database schema...")
                cursor.execute('''
                    CREATE TABLE users_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(email)
                    )
                ''')

                # Copy existing data
                cursor.execute('''
                    INSERT INTO users_new (id, name, email, created_at, updated_at)
                    SELECT id, name, email, created_at, updated_at FROM users
                ''')

                # Drop old table and rename new one
                cursor.execute('DROP TABLE users')
                cursor.execute('ALTER TABLE users_new RENAME TO users')

            elif not users_table:
                # Table doesn't exist, create it
                cursor.execute('''
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(email)
                    )
                ''')

            # Face analyses table - store analysis results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    image_path TEXT NOT NULL,
                    analysis_type TEXT NOT NULL, -- 'analyze' or 'verify'
                    result_data TEXT NOT NULL, -- JSON string
                    confidence_score REAL,
                    processing_time REAL,
                    model_used TEXT,
                    detector_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # Face embeddings table - store face embeddings for comparison
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    embedding_data TEXT NOT NULL, -- JSON string of numpy array
                    face_location TEXT, -- JSON string of facial area coordinates
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES face_analyses (id)
                )
            ''')

            # Verification history table - store verification comparisons
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS verification_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image1_id INTEGER,
                    image2_id INTEGER,
                    similarity_score REAL NOT NULL,
                    verified BOOLEAN NOT NULL,
                    threshold_used REAL,
                    model_used TEXT,
                    detector_used TEXT,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image1_id) REFERENCES face_analyses (id),
                    FOREIGN KEY (image2_id) REFERENCES face_analyses (id)
                )
            ''')

            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON face_analyses(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analyses_type ON face_analyses(analysis_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analyses_created ON face_analyses(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_analysis ON face_embeddings(analysis_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_verification_created ON verification_history(created_at)')

            conn.commit()

    def add_user(self, name: str, email: Optional[str] = None) -> int:
        """Add a new user to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                # Try to insert new user
                cursor.execute('''
                    INSERT INTO users (name, email)
                    VALUES (?, ?)
                ''', (name, email))
                conn.commit()
                return cursor.lastrowid

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e) and email:
                    # Email already exists, try to find existing user
                    cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
                    existing_user = cursor.fetchone()

                    if existing_user:
                        # Check if we should update the name
                        cursor.execute('SELECT name FROM users WHERE id = ?', (existing_user[0],))
                        current_name = cursor.fetchone()[0]

                        # Update name if different
                        if current_name != name:
                            cursor.execute('''
                                UPDATE users SET name = ?, updated_at = CURRENT_TIMESTAMP
                                WHERE id = ?
                            ''', (name, existing_user[0]))
                            conn.commit()

                        return existing_user[0]
                    else:
                        # This shouldn't happen, but handle it gracefully
                        return self.add_user(name, None)  # Try without email
                else:
                    # Re-raise other integrity errors
                    raise

    def save_analysis(self, user_id: Optional[int], image_path: str, analysis_type: str,
                     result_data: Dict[str, Any], confidence_score: Optional[float] = None,
                     processing_time: Optional[float] = None, model_used: Optional[str] = None,
                     detector_used: Optional[str] = None) -> int:
        """Save face analysis results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Convert NumPy types to JSON-serializable types
            cleaned_result_data = self.numpy_to_python_types(result_data)

            cursor.execute('''
                INSERT INTO face_analyses
                (user_id, image_path, analysis_type, result_data, confidence_score, processing_time, model_used, detector_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, image_path, analysis_type, json.dumps(cleaned_result_data), confidence_score,
                  processing_time, model_used, detector_used))
            analysis_id = cursor.lastrowid
            conn.commit()
            return analysis_id

    def save_embedding(self, analysis_id: int, embedding_data: np.ndarray,
                      face_location: Optional[Dict[str, Any]] = None) -> int:
        """Save face embedding data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Convert NumPy array to list for JSON serialization
            embedding_list = self.numpy_to_python_types(embedding_data)
            cleaned_face_location = self.numpy_to_python_types(face_location) if face_location else None

            cursor.execute('''
                INSERT INTO face_embeddings (analysis_id, embedding_data, face_location)
                VALUES (?, ?, ?)
            ''', (analysis_id, json.dumps(embedding_list),
                  json.dumps(cleaned_face_location) if cleaned_face_location else None))
            embedding_id = cursor.lastrowid
            conn.commit()
            return embedding_id

    def save_verification(self, image1_id: int, image2_id: int, similarity_score: float,
                         verified: bool, threshold_used: float, model_used: Optional[str] = None,
                         detector_used: Optional[str] = None, processing_time: Optional[float] = None) -> int:
        """Save verification comparison results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Convert NumPy types to Python types
            similarity_score = float(similarity_score) if hasattr(similarity_score, 'item') else similarity_score
            verified = bool(verified) if hasattr(verified, 'item') else verified
            threshold_used = float(threshold_used) if hasattr(threshold_used, 'item') else threshold_used
            processing_time = float(processing_time) if processing_time and hasattr(processing_time, 'item') else processing_time

            cursor.execute('''
                INSERT INTO verification_history
                (image1_id, image2_id, similarity_score, verified, threshold_used, model_used, detector_used, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (image1_id, image2_id, similarity_score, verified, threshold_used,
                  model_used, detector_used, processing_time))
            verification_id = cursor.lastrowid
            conn.commit()
            return verification_id

    def get_user_analyses(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all analyses for a specific user"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM face_analyses
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (user_id, limit))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result['result_data'] = json.loads(result['result_data'])
                results.append(result)

            return results

    def get_analysis_by_id(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """Get analysis by ID with user information"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT fa.*, COALESCE(u.name, 'Anonymous') as user_name, u.email as user_email
                FROM face_analyses fa
                LEFT JOIN users u ON fa.user_id = u.id
                WHERE fa.id = ?
            ''', (analysis_id,))

            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['result_data'] = json.loads(result['result_data'])
                # Ensure user_name is never None or 'None' string
                if not result.get('user_name') or result['user_name'] == 'None':
                    result['user_name'] = 'Anonymous'
                return result
            return None

    def get_embedding_by_analysis_id(self, analysis_id: int) -> Optional[np.ndarray]:
        """Get face embedding by analysis ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM face_embeddings WHERE analysis_id = ?', (analysis_id,))

            row = cursor.fetchone()
            if row:
                embedding_data = json.loads(row['embedding_data'])
                return np.array(embedding_data)
            return None

    def get_verification_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent verification history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT vh.*, a1.image_path as image1_path, a2.image_path as image2_path
                FROM verification_history vh
                LEFT JOIN face_analyses a1 ON vh.image1_id = a1.id
                LEFT JOIN face_analyses a2 ON vh.image2_id = a2.id
                ORDER BY vh.created_at DESC
                LIMIT ?
            ''', (limit,))

            results = []
            for row in cursor.fetchall():
                results.append(dict(row))

            return results

    def search_similar_faces(self, target_embedding: np.ndarray, threshold: float = 0.6,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar faces using embedding comparison"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM face_embeddings')

            results = []
            target_embedding = target_embedding.flatten()

            for row in cursor.fetchall():
                stored_embedding = np.array(json.loads(row['embedding_data'])).flatten()

                # Calculate cosine similarity
                similarity = np.dot(target_embedding, stored_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(stored_embedding)
                )

                if similarity >= threshold:
                    # Get the associated analysis data
                    analysis_id = row['analysis_id']
                    analysis = self.get_analysis_by_id(analysis_id)
                    if analysis:
                        # Check if image file exists
                        image_path = analysis.get('image_path', '')
                        if image_path and os.path.exists(image_path):
                            results.append({
                                'analysis': analysis,
                                'embedding': stored_embedding,
                                'similarity': float(similarity)
                            })
                        else:
                            # Skip results where image files don't exist
                            pass

            # Sort by similarity (highest first)
            results.sort(key=lambda x: x['similarity'], reverse=True)

            return results[:limit]

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get counts
            cursor.execute('SELECT COUNT(*) as user_count FROM users')
            user_count = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) as analysis_count FROM face_analyses')
            analysis_count = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) as embedding_count FROM face_embeddings')
            embedding_count = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) as verification_count FROM verification_history')
            verification_count = cursor.fetchone()[0]

            # Get recent activity
            cursor.execute('''
                SELECT COUNT(*) as recent_analyses
                FROM face_analyses
                WHERE created_at >= datetime('now', '-7 days')
            ''')
            recent_analyses = cursor.fetchone()[0]

            return {
                'total_users': user_count,
                'total_analyses': analysis_count,
                'total_embeddings': embedding_count,
                'total_verifications': verification_count,
                'recent_analyses_7days': recent_analyses,
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            }

    def export_data(self, export_path: str, data_type: str = "all"):
        """Export database data to JSON file"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'database_stats': self.get_database_stats()
            }

            if data_type in ["all", "users"]:
                cursor.execute('SELECT * FROM users')
                export_data['users'] = [dict(row) for row in cursor.fetchall()]

            if data_type in ["all", "analyses"]:
                cursor.execute('SELECT * FROM face_analyses')
                analyses = []
                for row in cursor.fetchall():
                    analysis = dict(row)
                    analysis['result_data'] = json.loads(analysis['result_data'])
                    analyses.append(analysis)
                export_data['analyses'] = analyses

            if data_type in ["all", "verifications"]:
                cursor.execute('''
                    SELECT vh.*, a1.image_path as image1_path, a2.image_path as image2_path
                    FROM verification_history vh
                    LEFT JOIN face_analyses a1 ON vh.image1_id = a1.id
                    LEFT JOIN face_analyses a2 ON vh.image2_id = a2.id
                ''')
                export_data['verifications'] = [dict(row) for row in cursor.fetchall()]

            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            return export_path

    def clear_database(self, confirm: bool = False):
        """Clear all data from database (requires confirmation)"""
        if not confirm:
            raise ValueError("Confirmation required. Set confirm=True to proceed.")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM verification_history')
            cursor.execute('DELETE FROM face_embeddings')
            cursor.execute('DELETE FROM face_analyses')
            cursor.execute('DELETE FROM users')
            conn.commit()

    def get_all_analyses(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all analyses with image paths"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT fa.*, COALESCE(u.name, 'Anonymous') as user_name, u.email as user_email
                FROM face_analyses fa
                LEFT JOIN users u ON fa.user_id = u.id
                ORDER BY fa.created_at DESC
                LIMIT ?
            ''', (limit,))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result['result_data'] = json.loads(result['result_data'])
                results.append(result)

            return results

    def get_analyses_with_images(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get analyses that have valid image files"""
        all_analyses = self.get_all_analyses(limit * 2)  # Get more to filter valid images
        valid_analyses = []

        for analysis in all_analyses:
            image_path = analysis.get('image_path')
            if image_path and os.path.exists(image_path):
                try:
                    # Quick check if image is readable
                    with Image.open(image_path) as img:
                        img.verify()
                    valid_analyses.append(analysis)
                except Exception:
                    continue  # Skip invalid images

            if len(valid_analyses) >= limit:
                break

        return valid_analyses
