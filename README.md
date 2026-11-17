# 🚗 Autonomous Driving Data Search

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

An advanced **multimodal semantic search engine** designed for autonomous vehicle (AV) and ADAS perception teams. Powered by **OpenAI CLIP (Vision Transformer)** and the **Qdrant Vector Database**, this system allows engineers to query raw, unannotated driving datasets using **natural-language descriptions** to instantly retrieve specific driving scenarios, edge cases, and safety-critical events.

---

## 📌 Why This Project?

Autonomous driving teams collect petabytes of drive logs every day. Finding critical edge cases — such as:
- *"A pedestrian with an umbrella crossing at night"*
- *"Emergency vehicle cutting in on a rainy highway"*
- *"Construction barrels blocking the right lane during sunset"*

traditionally requires manual metadata tagging or hours of scrubbing through footage. 

**Autonomous Driving Data Search** eliminates manual tagging. By embedding both text and imagery into a shared 512-dimensional vector space using **CLIP**, it enables **zero-shot cross-modal retrieval**: describe any driving condition in plain English, and the system retrieves the most semantically relevant frames instantly with mathematical cosine similarity confidence scores.

---

## 🏗️ System Architecture

```
                                      [ INGESTION PIPELINE ]
                                      
  ┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
  │  Drive Dataset  │ ───►  │ CLIP Image Encoder     │ ───►  │ Qdrant Vector DB     │
  │ (data/*.jpg/png)│       │ ViT-B/32 + L2 Norm     │       │ (Cosine Distance)    │
  └─────────────────┘       └────────────────────────┘       └──────────────────────┘
                                                                        ▲
                                                                        │ 512-d Cosine
                                                                        │ Similarity Search
                                      [ QUERY PIPELINE ]                │
                                                                        ▼
  ┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
  │ User Search     │ ───►  │ FastAPI Backend        │ ───►  │ Top-K Scenario       │
  │ Streamlit UI    │       │ CLIP Text Encoder      │       │ Image Matches        │
  └─────────────────┘       └────────────────────────┘       └──────────────────────┘
```

---

## ✨ Key Features

- **🧠 Zero-Shot Multimodal Retrieval:** Query vision datasets with open-vocabulary text without needing prior class labels or manual annotations.
- **⚡ Hardware Acceleration:** Native out-of-the-box support for **Apple Silicon (MPS)**, **NVIDIA GPUs (CUDA)**, and automatic fallback to CPU.
- **🚀 High-Performance Vector Storage:** Backed by **Qdrant** for sub-millisecond approximate nearest neighbor (HNSW / Cosine) vector search.
- **📦 Batched Ingestion Pipeline:** `populate_db.py` processes image datasets in parallel batches with progress tracking and performance metrics.
- **🌐 Production-Ready FastAPI Backend:** Async lifespan management, automated OpenAPI/Swagger documentation, CORS support, and health monitoring endpoints.
- **🎨 Interactive Streamlit Interface:** 
  - Dynamic result counts and confidence threshold sliders
  - Real-time connection and vector database health monitoring
  - Color-coded similarity confidence badges
  - Multi-column responsive layout

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Vision-Language Model** | [OpenAI CLIP](https://github.com/openai/CLIP) (`clip-vit-base-patch32`) | Embeds images & text into a shared 512-d semantic space |
| **Vector Database** | [Qdrant](https://qdrant.tech/) | High-scale vector database running via Docker |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | High-performance ASGI REST API |
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Interactive dashboard with real-time scenario preview |
| **Deep Learning** | PyTorch + Hugging Face Transformers | Tensor operations, normalization & inference |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+**
- **Docker Desktop** (for Qdrant vector database)

### 2. Clone the Repository
```bash
git clone https://github.com/mohammedkhadirkhadrijk8-max/Autonomous-Driving-Data-Search.git
cd Autonomous-Driving-Data-Search
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Qdrant Vector Database
Make sure Docker Desktop is running, then execute:
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
Verify Qdrant is live at [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

### 5. Add Images & Ingest Embeddings
Place your autonomous driving imagery (JPEG, PNG, WebP) into the `data/` folder:
```bash
# Ingest and index image embeddings into Qdrant
python populate_db.py
```

### 6. Start the FastAPI Backend
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API docs will be accessible at:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 7. Launch the Streamlit Web Application
In a separate terminal:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📡 REST API Reference

### Health & Diagnostics
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "device": "mps",
  "model": "openai/clip-vit-base-patch32",
  "indexed_images": 1
}
```

### Semantic Search
```http
GET /search?text_query={query}&limit={limit}
```
**Parameters:**
- `text_query` *(string, required)*: Natural language description (e.g. `"red sports car on highway"`)
- `limit` *(integer, optional)*: Maximum number of matches to return (default: `5`, min: `1`, max: `50`)

**Response:**
```json
[
  {
    "filename": "car.jpg",
    "score": 0.2506
  }
]
```

---

## 📂 Project Structure

```
Autonomous-Driving-Data-Search/
├── app.py              # Streamlit frontend with search interface & health monitor
├── main.py             # FastAPI backend exposing search & health endpoints
├── populate_db.py      # Batch embedding ingestion pipeline for Qdrant
├── requirements.txt    # Project Python dependencies
├── .gitignore          # Git ignore specifications
├── data/               # Autonomous driving dataset images (.jpg, .png, etc.)
│   └── car.jpg         # Sample driving frame
└── README.md           # Project documentation and architecture guide
```

---

## 🔮 Roadmap

- [ ] Support for public AV benchmark datasets (**nuScenes**, **KITTI**, **Waymo Open Dataset**)
- [ ] Temporal multi-frame / video clip segment retrieval
- [ ] Bounding-box and crop-level object semantic search
- [ ] Multimodal query refinement (combine text query + reference image)

---

## 📄 License

This project is licensed under the Apache 2.0 License.
