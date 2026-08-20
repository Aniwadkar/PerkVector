"""Build FAISS vector database from card JSON"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.card_loader import CardLoader
from src.data.text_chunker import CardTextChunker
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from src.config import CARDS_JSON_PATH, VECTOR_DB_PATH

def build_vector_db():
    """Build and save vector database"""
    print("=" * 60)
    print("Building PerkVector Vector Database")
    print("=" * 60)
    
    # Step 1: Load cards
    print("\n[1/4] Loading credit cards...")
    loader = CardLoader(CARDS_JSON_PATH)
    cards = loader.load_cards()
    print(f"✅ Loaded {len(cards)} cards")
    
    # Step 2: Create text chunks
    print("\n[2/4] Creating text chunks...")
    chunker = CardTextChunker()
    chunks = chunker.create_chunks(cards)
    print(f"✅ Created {len(chunks)} text chunks")
    
    # Print sample chunk
    print("\n📝 Sample chunk:")
    print(f"Card: {chunks[0]['card_name']}")
    print(f"Text: {chunks[0]['text'][:200]}...")
    
    # Step 3: Generate embeddings
    print("\n[3/4] Generating embeddings...")
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_chunks(chunks)
    print(f"✅ Generated embeddings with shape {embeddings.shape}")
    
    # Step 4: Build and save FAISS index
    print("\n[4/4] Building FAISS index...")
    vector_store = VectorStore()
    vector_store.build_index(embeddings, chunks)
    vector_store.save(VECTOR_DB_PATH)
    
    print("\n" + "=" * 60)
    print(f"✅ Vector database successfully saved to {VECTOR_DB_PATH}")
    print("=" * 60)
    
    # Test the index
    print("\n🧪 Testing vector search...")
    test_query = "cards with airport lounge access"
    print(f"Query: '{test_query}'")
    
    query_embedding = embedder.embed_text(test_query)
    results = vector_store.search(query_embedding, k=3)
    
    print(f"\nTop 3 results:")
    for i, (chunk, distance) in enumerate(results, 1):
        print(f"  {i}. {chunk['card_name']} (distance: {distance:.4f})")
    
    print("\n✅ All done! Vector database is ready to use.")

if __name__ == "__main__":
    build_vector_db()
