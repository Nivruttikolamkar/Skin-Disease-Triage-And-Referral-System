import sqlite3
import os

def migrate_database():
    db_paths = ["instance/database.db", "database.db"]
    for db_path in db_paths:
        if os.path.exists(db_path):
            print(f"Checking database at: {db_path}")
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            
            # Check existing columns in 'case' table
            cur.execute("PRAGMA table_info('case')")
            columns = [col[1] for col in cur.fetchall()]
            
            if columns and "city" not in columns:
                print(f"Adding missing 'city' column to {db_path}...")
                cur.execute('ALTER TABLE "case" ADD COLUMN city VARCHAR(100)')
                conn.commit()
                print("Migration successful! Column 'city' added.")
            elif columns:
                print(f"Table 'case' in {db_path} already has 'city' column.")
            
            conn.close()

if __name__ == "__main__":
    migrate_database()
