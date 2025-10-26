#!/usr/bin/env python3
"""
Check database contents to debug search issues
"""

import sqlite3
import json
import os

def check_database():
    """Check database contents"""
    db_path = "face_database.db"

    if not os.path.exists(db_path):
        print("❌ Database file does not exist!")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"📊 Database Tables: {[table[0] for table in tables]}")

            # Check analyses count
            cursor.execute("SELECT COUNT(*) FROM face_analyses")
            analyses_count = cursor.fetchone()[0]
            print(f"📋 Total Analyses: {analyses_count}")

            # Check embeddings count
            cursor.execute("SELECT COUNT(*) FROM face_embeddings")
            embeddings_count = cursor.fetchone()[0]
            print(f"🧠 Total Embeddings: {embeddings_count}")

            # Check users count
            cursor.execute("SELECT COUNT(*) FROM users")
            users_count = cursor.fetchone()[0]
            print(f"👥 Total Users: {users_count}")

            # Check verification history
            cursor.execute("SELECT COUNT(*) FROM verification_history")
            verifications_count = cursor.fetchone()[0]
            print(f"✅ Total Verifications: {verifications_count}")

            if analyses_count > 0:
                print("\n📋 Recent Analyses:")
                cursor.execute('''
                    SELECT fa.id, fa.user_id, fa.image_path, fa.analysis_type, fa.created_at, u.name as user_name
                    FROM face_analyses fa
                    LEFT JOIN users u ON fa.user_id = u.id
                    ORDER BY fa.created_at DESC
                    LIMIT 5
                ''')

                for row in cursor.fetchall():
                    print(f"  ID: {row[0]}, User: {row[5] or 'Anonymous'}, Image: {os.path.basename(row[2])}, Type: {row[3]}, Date: {row[4]}")

            if embeddings_count > 0:
                print("\n🧠 Face Embeddings:")
                cursor.execute('''
                    SELECT fe.id, fe.analysis_id, fa.image_path, fa.analysis_type
                    FROM face_embeddings fe
                    JOIN face_analyses fa ON fe.analysis_id = fa.id
                    ORDER BY fe.created_at DESC
                    LIMIT 5
                ''')

                for row in cursor.fetchall():
                    print(f"  Embedding ID: {row[0]}, Analysis ID: {row[1]}, Image: {os.path.basename(row[2])}, Type: {row[3]}")

                    # Check if embedding data is valid JSON
                    cursor.execute("SELECT embedding_data FROM face_embeddings WHERE id = ?", (row[0],))
                    embedding_data = cursor.fetchone()[0]
                    try:
                        parsed = json.loads(embedding_data)
                        print(f"    ✅ Embedding data: Valid JSON, length: {len(parsed)}")
                    except json.JSONDecodeError as e:
                        print(f"    ❌ Embedding data: Invalid JSON - {e}")

            print("\n🔍 Database check complete!")
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == '__main__':
    check_database()
