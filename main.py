import torch
from fastapi import FastAPI, HTTPException, Query
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

# 1. Hardware configuration for Apple Silicon (MPS)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 2. Initialize CLIP model and processor
model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

model.eval()

# 3. Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

collection_name = "autonomous_driving_data"

# 4. Initialize FastAPI
app = FastAPI(title="Autonomous Driving Search Engine")


# 5. Search endpoint
@app.get("/search")
def search_images(
    text_query: str = Query(..., min_length=1),
    limit: int = Query(default=3, ge=1, le=50)
):
    try:
        # Process text query through CLIP
        inputs = processor(
            text=[text_query],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        # Generate text embeddings
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        # FIX: Handle different Transformers output formats
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output

        # Normalize embeddings for cosine similarity
        text_features = text_features / text_features.norm(
            p=2, dim=-1, keepdim=True
        )

        # Convert tensor to a flat Python list
        search_vector = (
            text_features.detach()
            .cpu()
            .numpy()[0]
            .astype(float)
            .tolist()
        )

        # Query Qdrant for closest matching image embeddings
        search_result = client.query_points(
            collection_name=collection_name,
            query=search_vector,
            limit=limit,
            with_payload=True
        ).points

        # Return filenames and similarity scores
        return [
            {
                "filename": hit.payload.get("filename", "Unknown"),
                "score": hit.score
            }
            for hit in search_result
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image search failed: {str(e)}"
        )