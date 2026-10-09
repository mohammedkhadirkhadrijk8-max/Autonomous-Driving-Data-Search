import streamlit as st
import requests
import os

st.set_page_config(layout="wide")
st.title("Autonomous Driving Data Search")
st.write("Query the vector database to retrieve specific driving scenarios.")

# Search bar
query = st.text_input("Search (e.g., 'a red car on a rainy road'):")

if query:
    # Send the text to your FastAPI backend
    try:
        response = requests.get(f"http://127.0.0.1:8000/search?text_query={query}&limit=3").json()


        
        st.subheader("Search Results")
        # Create columns to display images side-by-side
        cols = st.columns(len(response))
        
        for idx, result in enumerate(response):
            with cols[idx]:
                image_path = os.path.join("data", result["filename"])
                # Display the image and the Qdrant similarity score
                st.image(image_path, caption=f"Confidence Score: {result['score']:.3f}", use_column_width=True)
                
    except Exception as e:
        st.error("Make sure your FastAPI backend is running!")
