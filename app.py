import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Enable CORS so your GitHub Pages frontend website can talk to this API safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from your GitHub Pages deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and DB in system RAM
encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(location=":memory:")
COLLECTION_NAME = "enterprise_docs"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = json.loads(contents.decode("utf-8"))
        
        documents = [item["text"] for item in data]
        metadata_payload = [{"category": item.get("category", "general"), "document": item["text"]} for item in data]
        generated_ids = list(range(1, len(documents) + 1))
        
        embeddings = encoder.encode(documents)
        vector_lists = [v.tolist() for v in embeddings]
        
        client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=vector_lists,
            payload=metadata_payload,
            ids=generated_ids
        )
        return {"status": "success", "message": f"Successfully indexed {len(documents)} chunks!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/search")
async def search_data(query: str, category: str = "All Categories"):
    query_filter = None
    if category and category != "All Categories":
        query_filter = models.Filter(
            must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))]
        )
    
    try:
        query_vector = encoder.encode(query).tolist()
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=3
        ).points
        
        results = []
        for hit in search_result:
            results.append({
                "score": round(hit.score, 4),
                "category": hit.payload.get("category", "general"),
                "text": hit.payload.get("document", "")
            })
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def home():
    return {"message": "Zero-LLM Search API is Live!"}
