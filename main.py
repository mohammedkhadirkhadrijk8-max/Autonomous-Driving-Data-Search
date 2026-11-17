import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "openai/clip-vit-base-patch32"
COLLECTION_NAME = "autonomous_driving_data"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Global references (populated at startup)
model = None
processor = None
device = None
qdrant_client = None


def get_device() -> torch.device:
    """Select the best available hardware accelerator."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# Application lifespan (startup / shutdown)
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connect to Qdrant once at startup."""
    global model, processor, device, qdrant_client

    device = get_device()
    print(f"🖥️  Device: {device}")

    print(f"⏳ Loading CLIP model '{MODEL_NAME}'...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    print("✅ Model loaded")

    print(f"🔗 Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Verify collection exists
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found. "
            "Run `python populate_db.py` first."
        )

    info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"✅ Connected — {info.points_count} vectors in '{COLLECTION_NAME}'")

    yield  # App runs here

    print("👋 Shutting down...")


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="Autonomous Driving Data Search API",
    description=(
        "Semantic image search engine for autonomous driving datasets. "
        "Uses OpenAI CLIP to match natural-language queries against "
        "pre-indexed driving scenario images stored in Qdrant."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Quick liveness / readiness probe."""
    info = qdrant_client.get_collection(COLLECTION_NAME)
    return {
        "status": "healthy",
        "device": str(device),
        "model": MODEL_NAME,
        "indexed_images": info.points_count,
    }


@app.get("/search")
def search_images(
    text_query: str = Query(..., min_length=1, description="Natural-language search query"),
    limit: int = Query(default=5, ge=1, le=50, description="Number of results to return"),
):
    """
    Search for driving scenario images by text description.

    The query is encoded by CLIP into an embedding and matched against
    pre-indexed image embeddings using cosine similarity in Qdrant.
    """
    try:
        # Encode the text query through CLIP
        inputs = processor(
            text=[text_query],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        # Handle Transformers ≥5.x output format
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output

        # L2 normalize for cosine similarity
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        search_vector = (
            text_features.detach().cpu().numpy()[0].astype(float).tolist()
        )

        # Query Qdrant
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=search_vector,
            limit=limit,
            with_payload=True,
        ).points

        return [
            {
                "filename": hit.payload.get("filename", "unknown"),
                "score": round(hit.score, 4),
            }
            for hit in search_result
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )
