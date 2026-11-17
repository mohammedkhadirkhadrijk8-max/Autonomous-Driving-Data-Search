import os
import sys
import time
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
COLLECTION_NAME = "autonomous_driving_data"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
IMAGE_DIR = os.getenv("IMAGE_DIR", "data")
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
BATCH_SIZE = 16  # Process images in batches for efficiency


def get_device() -> torch.device:
    """Select the best available hardware accelerator."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device: torch.device):
    """Load the CLIP model and processor."""
    print(f"⏳ Loading CLIP model '{MODEL_NAME}'...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"✅ Model loaded on {device}")
    return model, processor


def discover_images(image_dir: str) -> list[str]:
    """Recursively discover all supported image files."""
    images = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if f.lower().endswith(SUPPORTED_EXTENSIONS):
                images.append(os.path.join(root, f))
    return images


def generate_embeddings(
    image_paths: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> list[PointStruct]:
    """Generate CLIP embeddings for a list of images in batches."""
    points = []
    total = len(image_paths)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_paths = image_paths[batch_start : batch_start + BATCH_SIZE]
        batch_images = []
        batch_names = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                # Store relative path from IMAGE_DIR for portability
                batch_names.append(os.path.relpath(path, IMAGE_DIR))
            except Exception as e:
                print(f"  ⚠️  Skipping {path}: {e}")
                continue

        if not batch_images:
            continue

        # Process the whole batch at once
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Handle Transformers ≥5.x output format
        if not isinstance(image_features, torch.Tensor):
            image_features = getattr(
                image_features, "image_embeds",
                getattr(image_features, "pooler_output", image_features),
            )

        # L2 normalize for cosine similarity
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        embeddings = image_features.cpu().numpy()

        for i, name in enumerate(batch_names):
            idx = batch_start + i
            points.append(
                PointStruct(
                    id=idx,
                    vector=embeddings[i].tolist(),
                    payload={"filename": name},
                )
            )

        processed = min(batch_start + len(batch_paths), total)
        print(f"  📸 Processed {processed}/{total} images")

    return points


def main():
    device = get_device()
    model, processor = load_model(device)

    # Connect to Qdrant
    print(f"🔗 Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Recreate collection (clean start each run)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")

    # Discover images
    os.makedirs(IMAGE_DIR, exist_ok=True)
    image_paths = discover_images(IMAGE_DIR)

    if not image_paths:
        print(f"\n❌ No images found in '{IMAGE_DIR}/'. Add images and re-run.")
        sys.exit(1)

    print(f"🔍 Found {len(image_paths)} images in '{IMAGE_DIR}/'")

    # Generate embeddings and upload
    start = time.time()
    points = generate_embeddings(image_paths, model, processor, device)
    elapsed = time.time() - start

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"\n🎉 Successfully indexed {len(points)} images in {elapsed:.1f}s")
        print(f"   Average: {elapsed / len(points):.2f}s per image")
    else:
        print("\n❌ No images could be processed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
