from utils import load_chroma_db

chroma_db = load_chroma_db()

collection = chroma_db._collection
print(f"ChromaDB'deki toplam vektör sayısı: {collection.count()}")
