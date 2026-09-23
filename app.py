import json
import os
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient, models


app = FastAPI()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = (
    "https://router.huggingface.co/"
    "hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/"
    "pipeline/feature-extraction"
)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embeddings(texts: list):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is not set on Render.")

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": texts},
        timeout=60
    )

    # Raise an actual error for 4xx/5xx responses
    response.raise_for_status()

    result = response.json()

    # Hugging Face can return an error object
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Hugging Face API Error: {result['error']}")

    return result


# ============================================================
# QDRANT
# ============================================================

# In-memory Qdrant for the demo
client = QdrantClient(location=":memory:")

COLLECTION_NAME = "enterprise_docs"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE
    )
)


# ============================================================
# UPLOAD
# ============================================================

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        data = json.loads(contents.decode("utf-8"))

        documents = [
            item["text"]
            for item in data
        ]

        metadata_payload = [
            {
                "category": item.get("category", "general"),
                "document": item["text"]
            }
            for item in data
        ]

        generated_ids = list(
            range(1, len(documents) + 1)
        )

        # Generate embeddings through Hugging Face
        vector_lists = get_embeddings(documents)

        client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=vector_lists,
            payload=metadata_payload,
            ids=generated_ids
        )

        return {
            "status": "success",
            "message": f"Successfully indexed {len(documents)} chunks!"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# SEARCH
# ============================================================

@app.get("/search")
async def search_data(
    query: str,
    category: str = "All Categories"
):

    try:
        query_filter = None

        if category and category != "All Categories":
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(
                            value=category
                        )
                    )
                ]
            )

        # Generate embedding for the search query
        query_vector_res = get_embeddings([query])

        query_vector = query_vector_res[0]

        # Search Qdrant
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
                "category": hit.payload.get(
                    "category",
                    "general"
                ),
                "text": hit.payload.get(
                    "document",
                    ""
                )
            })

        return {
            "status": "success",
            "results": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Zero-LLM Search API is Live!"
    }
