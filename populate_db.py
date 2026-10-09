import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# 1. Hardware Config (Apple Silicon MPS)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 2. Load CLIP Model and Processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 3. Connect to Qdrant
client = QdrantClient("localhost", port=6333)
collection_name = "autonomous_driving_data"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

# 4. Generate Embeddings for Images
image_dir = "data"
os.makedirs(image_dir, exist_ok=True)
images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

points = []
for idx, image_name in enumerate(images):
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path).convert("RGB")
    
    # Process image through CLIP
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # FIX: Handle different Transformers output formats
    if not isinstance(image_features, torch.Tensor):
        image_features = getattr(image_features, "image_embeds", getattr(image_features, "pooler_output", image_features))
    
    # Convert tensor to a flat Python list
    embedding = image_features.cpu().numpy()[0].tolist()
    
    # Create Qdrant Point
    points.append(PointStruct(
        id=idx,
        vector=embedding,
        payload={"filename": image_name}
    ))

# 5. Upload to Database
if points:
    client.upsert(collection_name=collection_name, points=points)
    print(f"Successfully indexed {len(points)} images!")
else:
    print("No images found in the 'data/' folder. Please add some.")
