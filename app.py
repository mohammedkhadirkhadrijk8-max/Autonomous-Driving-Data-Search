import streamlit as st
import requests
import os

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomous Driving Data Search",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    num_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
    min_score = st.slider("Min confidence score", 0.0, 1.0, 0.0, 0.01)

    st.divider()

    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"Backend connected")
        st.caption(f"**Model:** `{health.get('model', 'N/A')}`")
        st.caption(f"**Device:** `{health.get('device', 'N/A')}`")
        st.caption(f"**Indexed images:** `{health.get('indexed_images', 0)}`")
    except Exception:
        st.error("⚠️ Backend offline")
        st.caption("Start it with: `uvicorn main:app`")

    st.divider()
    st.markdown(
        "Built with [CLIP](https://openai.com/research/clip) · "
        "[Qdrant](https://qdrant.tech) · "
        "[Streamlit](https://streamlit.io)"
    )

# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.title("🚗 Autonomous Driving Data Search")
st.markdown(
    "Search through driving scenario images using **natural language**. "
    "Powered by OpenAI CLIP for semantic understanding."
)

# Search bar
query = st.text_input(
    "🔍 Describe a driving scenario:",
    placeholder="e.g., 'a red car on a rainy highway at night'",
)

if query:
    with st.spinner("Searching..."):
        try:
            response = requests.get(
                f"{API_URL}/search",
                params={"text_query": query, "limit": num_results},
                timeout=30,
            )
            response.raise_for_status()
            results = response.json()

            # Handle FastAPI error responses
            if isinstance(results, dict) and "detail" in results:
                st.error(f"Backend error: {results['detail']}")
            else:
                # Apply score filter
                results = [r for r in results if r["score"] >= min_score]

                if not results:
                    st.warning("No results matched your query. Try a different description.")
                else:
                    st.subheader(f"📷 {len(results)} Result{'s' if len(results) != 1 else ''}")

                    # Display in a responsive grid (max 4 columns)
                    cols_per_row = min(len(results), 4)
                    for row_start in range(0, len(results), cols_per_row):
                        row_items = results[row_start : row_start + cols_per_row]
                        cols = st.columns(cols_per_row)

                        for idx, result in enumerate(row_items):
                            with cols[idx]:
                                image_path = os.path.join("data", result["filename"])
                                score = result["score"]

                                if os.path.exists(image_path):
                                    st.image(image_path, width="stretch")
                                    # Color-coded confidence badge
                                    if score >= 0.3:
                                        st.success(f"✅ Score: **{score:.3f}**")
                                    elif score >= 0.2:
                                        st.info(f"ℹ️ Score: **{score:.3f}**")
                                    else:
                                        st.warning(f"⚠️ Score: **{score:.3f}**")
                                    st.caption(f"`{result['filename']}`")
                                else:
                                    st.error(f"Image not found: `{result['filename']}`")

        except requests.exceptions.ConnectionError:
            st.error(
                "⚠️ Cannot connect to the backend! "
                "Make sure it's running with: `uvicorn main:app`"
            )
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The backend may be overloaded.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
else:
    # Show example queries when idle
    st.markdown("---")
    st.markdown("**💡 Try these example queries:**")
    example_cols = st.columns(3)
    examples = [
        "🌧️ Car driving in heavy rain",
        "🌙 Night-time highway driving",
        "🚶 Pedestrian crossing the street",
    ]
    for col, example in zip(example_cols, examples):
        col.info(example)
