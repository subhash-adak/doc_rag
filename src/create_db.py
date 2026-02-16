"""
Script to initialize the database and create tables
Run this before starting the application
"""

from src.app.v1.database.connection import engine, Base
from src.app.v1.models.database import User, Document, DocumentChunk, QueryHistory, UserStats

def init_db():
    print("🔧 Creating database tables...")
    
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
        print("\nTables created:")
        print("  - users")
        print("  - documents")
        print("  - document_chunks")
        print("  - query_history")
        print("  - user_stats")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        raise

# if __name__ == "__main__":
#     init_db()