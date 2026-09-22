import json
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient, models

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for Hugging Face's Free Serverless API
# ⚠️ PASTE YOUR HUGGING FACE TOKEN HERE (e.g., "hf_abcdef...")
HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN"
API_URL = "https://huggingface.co"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Helper function to get vectors without using local RAM
def get_embeddings(texts: list):
    response = requests.post(API_URL, headers=headers, json={"inputs": texts})
    return response.json()

# Initialize Qdrant in system RAM
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
        
        # Call Hugging Face API to get the embeddings (RAM stays at zero!)
        vector_lists = get_embeddings(documents)
        
        # If the API returned an error dictionary instead of a list
        if isinstance(vector_lists, dict) and "error" in vector_lists:
            return {"status": "error", "message": f"HF API Error: {vector_lists['error']}"}

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
        # Get query vector from Hugging Face API
        query_vector_res = get_embeddings([query])
        query_vector = query_vector_res[0]
        
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
