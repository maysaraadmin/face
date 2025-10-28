import sqlite3
import json
import os
import logging
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDatabase:
    """Database manager for storing face analysis results and user data"""

    def __init__(self, db_path: str = "face_database.db"):
        """Initialize the face database
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = str(Path(db_path).absolute())
        self._connection_pool = []
        self._max_connections = 5
        self._connection_timeout = 30  # seconds
        self._last_connection_time = 0
        self._init_connection_pool()
        self.init_database()
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection from the pool"""
        current_time = time.time()
        
        # Clean up old connections
        if current_time - self._last_connection_time > 30:  # Cleanup every 30 seconds
            self._cleanup_connections()
            self._last_connection_time = current_time
            
        # Try to get an existing connection
        while self._connection_pool:
            conn, last_used = self._connection_pool.pop()
            try:
                # Check if connection is still valid
                conn.execute('SELECT 1')
                return conn
            except (sqlite3.Error, sqlite3.ProgrammingError):
                continue
                
        # If no valid connection found, create a new one
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                isolation_level=None,  # Use autocommit mode
                check_same_thread=False  # Allow multiple threads to use the connection
            )
            # Enable WAL mode for better concurrency
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA cache_size=-2000')  # 2MB cache
            return conn
        except Exception as e:
            logger.error(f"Failed to create database connection: {str(e)}")
            raise
            
    def _return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool"""
        if len(self._connection_pool) < self._max_connections:
            self._connection_pool.append((conn, time.time()))
        else:
            try:
                conn.close()
            except:
                pass
                
    def _cleanup_connections(self):
        """Clean up old connections from the pool"""
        current_time = time.time()
        new_pool = []
        
        for conn, last_used in self._connection_pool:
            try:
                if current_time - last_used < self._connection_timeout:
                    # Check if connection is still valid
                    conn.execute('SELECT 1')
                    new_pool.append((conn, last_used))
                else:
                    conn.close()
            except:
                try:
                    conn.close()
                except:
                    pass
                    
        self._connection_pool = new_pool

    def close(self):
        """Close all database connections"""
        for conn, _ in self._connection_pool:
            try:
                conn.close()
            except:
                pass
        self._connection_pool = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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

    def _init_connection_pool(self):
        """Initialize the connection pool"""
        # Create the database directory if it doesn't exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        # Create the database file if it doesn't exist
        if not os.path.exists(self.db_path):
            open(self.db_path, 'a').close()
            
    def init_database(self):
        """Initialize database tables and indexes"""
        conn = None
        try:
            conn = self._get_connection()
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
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {str(e)}")
            if conn:
                conn.rollback()
            raise
            
        finally:
            if conn:
                self._return_connection(conn)

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

    def _cosine_similarity(self, vec1_json: str, vec2_json: str) -> float:
        """Calculate cosine similarity between two JSON-serialized vectors"""
        try:
            vec1 = np.array(json.loads(vec1_json))
            vec2 = np.array(json.loads(vec2_json))
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
            
    def _l2_distance(self, vec1_json: str, vec2_json: str) -> float:
        """Calculate L2 distance between two JSON-serialized vectors"""
        try:
            vec1 = np.array(json.loads(vec1_json))
            vec2 = np.array(json.loads(vec2_json))
            return float(np.linalg.norm(vec1 - vec2))
        except Exception as e:
            logger.error(f"Error calculating L2 distance: {str(e)}")
            return float('inf')
            
    def _calculate_embedding_hash(self, embedding_json: str) -> str:
        """Calculate a hash of the embedding for deduplication"""
        return hashlib.md5(embedding_json.encode()).hexdigest()
        
    def _image_exists(self, image_path: str) -> bool:
        """Check if an image file exists"""
        return os.path.exists(image_path)

    def save_analysis(self, user_id: Optional[int], image_path: str, analysis_type: str,
                     result_data: Dict[str, Any], confidence_score: Optional[float] = None,
                     processing_time: Optional[float] = None, model_used: Optional[str] = None,
                     detector_used: Optional[str] = None, metadata: Optional[Dict] = None) -> int:
        """Save face analysis results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Convert NumPy types to JSON-serializable types
            cleaned_result_data = self.numpy_to_python_types(result_data)

            # Add metadata to result data
            if metadata:
                if 'metadata' not in cleaned_result_data:
                    cleaned_result_data['metadata'] = {}
                cleaned_result_data['metadata'].update(metadata)
                
            # Add hash of the image for deduplication
            image_hash = None
            try:
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as f:
                        image_hash = hashlib.md5(f.read()).hexdigest()
            except Exception as e:
                logger.warning(f"Could not calculate image hash: {str(e)}")
            
            cursor.execute('''
                INSERT INTO face_analyses
                (user_id, image_path, image_hash, analysis_type, result_data, 
                 confidence_score, processing_time, model_used, detector_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, image_path, image_hash, analysis_type, 
                  json.dumps(cleaned_result_data), confidence_score,
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

    def search_similar_faces(self, target_embedding: np.ndarray, 
                           threshold: float = 0.6,
                           limit: int = 10,
                           user_id: Optional[int] = None,
                           min_confidence: Optional[float] = None,
                           max_results: int = 1000) -> List[Dict[str, Any]]:
        """Search for similar faces using efficient vector similarity search
        
        Args:
            target_embedding: The embedding vector to search with
            threshold: Minimum similarity score (0-1)
            limit: Maximum number of results to return
            user_id: Optional user ID to filter results
            min_confidence: Optional minimum confidence score for the analysis
            max_results: Maximum number of results to consider (for performance)
            
        Returns:
            List of matching faces with similarity scores
        """
        conn = None
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Convert target embedding to JSON for SQL function
            target_embedding_json = json.dumps(target_embedding.flatten().tolist())
            
            # Build the query
            query = '''
                SELECT 
                    fa.id as analysis_id,
                    fa.user_id,
                    fa.image_path,
                    fa.confidence_score,
                    fa.created_at,
                    fe.embedding_data,
                    cosine_similarity(?, fe.embedding_data) as similarity
                FROM face_embeddings fe
                JOIN face_analyses fa ON fe.analysis_id = fa.id
                WHERE 
                    cosine_similarity(?, fe.embedding_data) >= ?
                    AND image_exists(fa.image_path) = 1
            '''
            
            params = [target_embedding_json, target_embedding_json, threshold]
            
            # Add optional filters
            if user_id is not None:
                query += ' AND fa.user_id = ?'
                params.append(user_id)
                
            if min_confidence is not None:
                query += ' AND fa.confidence_score >= ?'
                params.append(min_confidence)
            
            # Order and limit
            query += ' ORDER BY similarity DESC LIMIT ?'
            params.append(min(limit * 2, max_results))  # Get extra for filtering
            
            # Execute the query
            cursor.execute(query, params)
            
            # Process results
            results = []
            for row in cursor.fetchall():
                try:
                    # Get the full analysis data
                    analysis = self.get_analysis_by_id(row['analysis_id'])
                    if not analysis:
                        continue
                        
                    # Add to results
                    results.append({
                        'analysis': analysis,
                        'embedding': np.array(json.loads(row['embedding_data'])),
                        'similarity': float(row['similarity'])
                    })
                    
                    # Early exit if we have enough high-quality matches
                    if len(results) >= limit and row['similarity'] > 0.9:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing search result: {str(e)}")
            
            # Sort and limit results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in search_similar_faces: {str(e)}")
            raise
            
        finally:
            if conn:
                self._return_connection(conn)

    def get_database_stats(self, detailed: bool = False) -> Dict[str, Any]:
        """Get database statistics
        
        Args:
            detailed: If True, include more detailed statistics
            
        Returns:
            Dictionary containing database statistics
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            stats = {}
            
            # Basic counts
            cursor.execute('SELECT COUNT(*) as user_count FROM users')
            stats['total_users'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) as analysis_count FROM face_analyses')
            stats['total_analyses'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) as embedding_count FROM face_embeddings')
            stats['total_embeddings'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) as verification_count FROM verification_history')
            stats['total_verifications'] = cursor.fetchone()[0]
            
            # Recent activity
            cursor.execute('''
                SELECT COUNT(*) as recent_analyses
                FROM face_analyses
                WHERE created_at >= datetime('now', '-7 days')
            ''')
            stats['recent_analyses_7days'] = cursor.fetchone()[0]
            
            # Database size
            stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            
            if detailed:
                # User activity
                cursor.execute('''
                    SELECT strftime('%Y-%m', created_at) as month, 
                           COUNT(*) as count
                    FROM face_analyses
                    GROUP BY month
                    ORDER BY month DESC
                    LIMIT 12
                ''')
                stats['analyses_by_month'] = [dict(row) for row in cursor.fetchall()]
                
                # Model usage
                cursor.execute('''
                    SELECT model_used, COUNT(*) as count
                    FROM face_analyses
                    WHERE model_used IS NOT NULL
                    GROUP BY model_used
                    ORDER BY count DESC
                ''')
                stats['models_used'] = [dict(row) for row in cursor.fetchall()]
                
                # Confidence distribution
                cursor.execute('''
                    SELECT 
                        COUNT(CASE WHEN confidence_score >= 0.9 THEN 1 END) as high_confidence,
                        COUNT(CASE WHEN confidence_score >= 0.7 AND confidence_score < 0.9 THEN 1 END) as medium_confidence,
                        COUNT(CASE WHEN confidence_score < 0.7 THEN 1 END) as low_confidence
                    FROM face_analyses
                    WHERE confidence_score IS NOT NULL
                ''')
                confidence_stats = cursor.fetchone()
                stats['confidence_distribution'] = {
                    'high': confidence_stats[0],
                    'medium': confidence_stats[1],
                    'low': confidence_stats[2]
                }
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            raise
            
        finally:
            if conn:
                self._return_connection(conn)

    def batch_import_analyses(self, analyses: List[Dict]) -> Dict[str, int]:
        """Batch import multiple face analyses
        
        Args:
            analyses: List of analysis dictionaries with required fields:
                     - image_path: Path to the image file
                     - embedding: Face embedding vector
                     - analysis_data: Dictionary with analysis results
                     - Optional: user_id, confidence_score, model_used, detector_used, metadata
                     
        Returns:
            Dictionary with import statistics
        """
        if not analyses:
            return {'total': 0, 'imported': 0, 'errors': 0, 'skipped': 0}
            
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Start transaction
            cursor.execute('BEGIN TRANSACTION')
            
            stats = {'total': len(analyses), 'imported': 0, 'errors': 0, 'skipped': 0}
            
            for i, analysis in enumerate(analyses, 1):
                try:
                    # Extract required fields
                    image_path = analysis.get('image_path')
                    embedding = analysis.get('embedding')
                    analysis_data = analysis.get('analysis_data', {})
                    
                    if not all([image_path, embedding is not None, analysis_data]):
                        stats['errors'] += 1
                        continue
                        
                    # Check if this image already exists
                    cursor.execute(
                        'SELECT id FROM face_analyses WHERE image_path = ?', 
                        (image_path,)
                    )
                    if cursor.fetchone():
                        stats['skipped'] += 1
                        continue
                    
                    # Save analysis
                    analysis_id = self.save_analysis(
                        user_id=analysis.get('user_id'),
                        image_path=image_path,
                        analysis_type=analysis.get('analysis_type', 'batch_import'),
                        result_data=analysis_data,
                        confidence_score=analysis.get('confidence_score'),
                        model_used=analysis.get('model_used'),
                        detector_used=analysis.get('detector_used'),
                        metadata=analysis.get('metadata')
                    )
                    
                    # Save embedding
                    if analysis_id and 'embedding' in analysis:
                        self.save_embedding(
                            analysis_id=analysis_id,
                            embedding_data=analysis['embedding'],
                            face_location=analysis.get('face_location')
                        )
                    
                    stats['imported'] += 1
                    
                    # Commit every 100 records
                    if i % 100 == 0:
                        conn.commit()
                        
                except Exception as e:
                    logger.error(f"Error importing analysis {i}: {str(e)}")
                    stats['errors'] += 1
                    # Continue with next record
            
            # Final commit
            conn.commit()
            return stats
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Batch import failed: {str(e)}")
            raise
            
        finally:
            if conn:
                self._return_connection(conn)

    def export_data(self, export_path: str, data_type: str = "all", 
                   format: str = "json", 
                   include_embeddings: bool = False) -> Dict[str, Any]:
        """Export database data to a file
        
        Args:
            export_path: Path to save the export file
            data_type: Type of data to export ('analyses', 'embeddings', 'users', 'all')
            format: Export format ('json' or 'sql')
            include_embeddings: Whether to include embedding vectors in the export
            
        Returns:
            Dictionary with export statistics
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            export_data = {
                'metadata': {
                    'exported_at': datetime.utcnow().isoformat(),
                    'database_version': '1.0',
                    'data_type': data_type,
                    'include_embeddings': include_embeddings
                },
                'data': {}
            }
            
            # Export data based on type
            if data_type in ('all', 'users'):
                cursor.execute('SELECT * FROM users')
                export_data['data']['users'] = [dict(row) for row in cursor.fetchall()]
                
            if data_type in ('all', 'analyses'):
                cursor.execute('SELECT * FROM face_analyses')
                analyses = []
                for row in cursor.fetchall():
                    analysis = dict(row)
                    # Parse JSON data
                    if 'result_data' in analysis and analysis['result_data']:
                        analysis['result_data'] = json.loads(analysis['result_data'])
                    analyses.append(analysis)
                export_data['data']['analyses'] = analyses
                
            if data_type in ('all', 'embeddings') and include_embeddings:
                cursor.execute('SELECT * FROM face_embeddings')
                export_data['data']['embeddings'] = [dict(row) for row in cursor.fetchall()]
                
            # Save to file
            os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
            
            if format.lower() == 'sql':
                # Export as SQL dump
                with open(export_path, 'w', encoding='utf-8') as f:
                    for line in conn.iterdump():
                        if not include_embeddings and 'face_embeddings' in line:
                            continue
                        f.write(f"{line}\n")
            else:
                # Default to JSON
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            # Return export stats
            stats = {
                'exported_at': export_data['metadata']['exported_at'],
                'file_path': os.path.abspath(export_path),
                'file_size_mb': os.path.getsize(export_path) / (1024 * 1024),
                'items_exported': {
                    'users': len(export_data['data'].get('users', [])),
                    'analyses': len(export_data['data'].get('analyses', [])),
                    'embeddings': len(export_data['data'].get('embeddings', []))
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            raise
            
        finally:
            if conn:
                self._return_connection(conn)
                
    def cleanup_orphaned_data(self, dry_run: bool = True) -> Dict[str, int]:
        """Clean up orphaned data (analyses without users, embeddings without analyses, etc.)
        
        Args:
            dry_run: If True, only report what would be deleted without making changes
            
        Returns:
            Dictionary with cleanup statistics
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            stats = {
                'orphaned_analyses': 0,
                'orphaned_embeddings': 0,
                'missing_images': 0,
                'duplicate_embeddings': 0,
                'total_records_processed': 0,
                'changes_made': 0
            }
            
            if not dry_run:
                cursor.execute('BEGIN TRANSACTION')
            
            try:
                # Find analyses with missing images
                cursor.execute('''
                    SELECT id, image_path 
                    FROM face_analyses 
                    WHERE image_exists(image_path) = 0
                ''')
                missing_images = cursor.fetchall()
                stats['missing_images'] = len(missing_images)
                
                if not dry_run and missing_images:
                    # Delete analyses with missing images
                    cursor.execute('''
                        DELETE FROM face_analyses 
                        WHERE id IN (SELECT id FROM face_analyses 
                                    WHERE image_exists(image_path) = 0)
                    ''')
                    stats['changes_made'] += cursor.rowcount
                
                # Find orphaned embeddings (no matching analysis)
                cursor.execute('''
                    SELECT fe.id 
                    FROM face_embeddings fe
                    LEFT JOIN face_analyses fa ON fe.analysis_id = fa.id
                    WHERE fa.id IS NULL
                ''')
                orphaned_embeddings = cursor.fetchall()
                stats['orphaned_embeddings'] = len(orphaned_embeddings)
                
                if not dry_run and orphaned_embeddings:
                    cursor.execute('''
                        DELETE FROM face_embeddings 
                        WHERE analysis_id IN (
                            SELECT fe.id 
                            FROM face_embeddings fe
                            LEFT JOIN face_analyses fa ON fe.analysis_id = fa.id
                            WHERE fa.id IS NULL
                        )
                    ''')
                    stats['changes_made'] += cursor.rowcount
                
                # Find duplicate embeddings (same image hash)
                cursor.execute('''
                    SELECT image_hash, COUNT(*) as count
                    FROM face_analyses
                    WHERE image_hash IS NOT NULL
                    GROUP BY image_hash
                    HAVING COUNT(*) > 1
                ''')
                duplicate_hashes = cursor.fetchall()
                
                for row in duplicate_hashes:
                    image_hash, count = row
                    stats['duplicate_embeddings'] += (count - 1)
                    
                    if not dry_run and count > 1:
                        # Keep the most recent analysis for each duplicate
                        cursor.execute('''
                            DELETE FROM face_analyses
                            WHERE id IN (
                                SELECT id FROM (
                                    SELECT id, 
                                           ROW_NUMBER() OVER (PARTITION BY image_hash ORDER BY created_at DESC) as rn
                                    FROM face_analyses
                                    WHERE image_hash = ?
                                ) t 
                                WHERE t.rn > 1
                            )
                        ''', (image_hash,))
                        stats['changes_made'] += cursor.rowcount
                
                if not dry_run:
                    conn.commit()
                    
                    # Vacuum to reclaim space
                    if stats['changes_made'] > 0:
                        cursor.execute('VACUUM')
                
                return stats
                
            except Exception as e:
                if not dry_run:
                    conn.rollback()
                raise
                
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise
            
        finally:
            if conn:
                self._return_connection(conn)

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

    def get_analyses(self, limit: int = 100, order_by: str = 'created_at DESC', **filters) -> List[Dict[str, Any]]:
        """Get analyses with optional filtering and ordering
        
        Args:
            limit: Maximum number of results to return
            order_by: SQL ORDER BY clause (e.g., 'created_at DESC')
            **filters: Key-value pairs to filter by (e.g., user_id=1)
            
        Returns:
            List of analysis dictionaries
        """
        conn = None
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build the query
            query = '''
                SELECT fa.*, 
                       COALESCE(u.name, 'Anonymous') as user_name, 
                       u.email as user_email
                FROM face_analyses fa
                LEFT JOIN users u ON fa.user_id = u.id
            '''
            
            # Add WHERE conditions if filters are provided
            params = []
            conditions = []
            for key, value in filters.items():
                if value is not None:
                    conditions.append(f"fa.{key} = ?")
                    params.append(value)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Add ORDER BY and LIMIT
            if order_by:
                query += f" ORDER BY {order_by}"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            # Execute query
            cursor.execute(query, params)
            
            # Process results
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if 'result_data' in result and result['result_data']:
                    result['result_data'] = json.loads(result['result_data'])
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error getting analyses: {str(e)}")
            raise
            
        finally:
            if conn:
                self._return_connection(conn)
                
    # Alias for backward compatibility
    def get_all_analyses(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all analyses (legacy method)"""
        return self.get_analyses(limit=limit)
        
    def get_analyses_with_images(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get analyses that have valid image files"""
        analyses = self.get_analyses(limit=limit * 2)  # Get more to filter valid images
        valid_analyses = []

        for analysis in analyses:
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
