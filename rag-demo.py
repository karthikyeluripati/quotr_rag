import streamlit as st
import json
import faiss
import openai
import numpy as np
import pandas as pd
import os
import pickle

# Set your OpenAI API key
openai.api_key = "your-openai-api-key"  # Replace with your actual API key

# File paths for storing data
COST_DB_FILE = "cost_database.json"
EMBEDDINGS_FILE = "embeddings.pkl"
INDEX_FILE = "faiss_index.pkl"

# Helper functions
def save_cost_database(data):
    with open(COST_DB_FILE, 'w') as f:
        json.dump(data, f)

def load_cost_database():
    if os.path.exists(COST_DB_FILE):
        with open(COST_DB_FILE, 'r') as f:
            return json.load(f)
    return []

def save_embeddings_and_index(embeddings, index):
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    faiss.write_index(index, INDEX_FILE)

def load_embeddings_and_index():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings = pickle.load(f)
        index = faiss.read_index(INDEX_FILE)
        return embeddings, index
    return None, None

def get_openai_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def find_best_match(query_embedding, index, cost_database):
    distances, indices = index.search(np.array([query_embedding]), 1)
    best_match_index = indices[0][0]
    return cost_database[best_match_index], distances[0][0]

# Page functions
def cost_database_page():
    st.title("Cost Database Management")

    # Load existing data
    cost_database = load_cost_database()

    # Display current database
    st.subheader("Current Cost Database")
    df = pd.DataFrame(cost_database)
    st.dataframe(df)

    # Add new item
    st.subheader("Add New Item")
    new_item = {}
    new_item['name'] = st.text_input("Item Name")
    new_item['description'] = st.text_area("Description")
    new_item['price'] = st.number_input("Price", min_value=0.0, format="%.2f")
    if st.button("Add Item"):
        cost_database.append(new_item)
        save_cost_database(cost_database)
        st.success("Item added successfully!")

    # Edit or delete item
    st.subheader("Edit or Delete Item")
    if cost_database:
        item_to_edit = st.selectbox("Select item to edit or delete", 
                                    range(len(cost_database)), 
                                    format_func=lambda i: cost_database[i]['name'])
        
        edit_item = cost_database[item_to_edit].copy()
        edit_item['name'] = st.text_input("Edit Name", edit_item['name'])
        edit_item['description'] = st.text_area("Edit Description", edit_item['description'])
        edit_item['price'] = st.number_input("Edit Price", value=float(edit_item['price']), format="%.2f")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Update Item"):
                cost_database[item_to_edit] = edit_item
                save_cost_database(cost_database)
                st.success("Item updated successfully!")
        with col2:
            if st.button("Delete Item"):
                del cost_database[item_to_edit]
                save_cost_database(cost_database)
                st.success("Item deleted successfully!")

    # Update embeddings
    if st.button("Update Embeddings"):
        with st.spinner("Updating embeddings..."):
            embeddings = [get_openai_embedding(f"{item['name']} {item['description']}") for item in cost_database]
            index = create_faiss_index(embeddings)
            save_embeddings_and_index(embeddings, index)
        st.success("Embeddings updated successfully!")

def canvas_data_page():
    st.title("Canvas Data Processing")

    # Load cost database and embeddings
    cost_database = load_cost_database()
    embeddings, index = load_embeddings_and_index()

    if not cost_database or embeddings is None or index is None:
        st.error("Please upload and process the cost database first.")
        return

    # File upload
    uploaded_file = st.file_uploader("Upload Canvas Data (JSON)", type="json")
    if uploaded_file is not None:
        canvas_data = json.load(uploaded_file)

        # Process canvas data
        results = []
        for item in canvas_data:
            query_embedding = get_openai_embedding(f"{item['name']} {item.get('description', '')}")
            best_match, distance = find_best_match(query_embedding, index, cost_database)
            results.append({
                "Canvas Item": item['name'],
                "Matched Item": best_match['name'],
                "Matched Description": best_match['description'],
                "Matched Price": best_match['price'],
                "Similarity Score": 1 / (1 + distance)
            })

        # Display results
        st.subheader("Matching Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="canvas_matching_results.csv",
            mime="text/csv",
        )

def main():
    st.sidebar.title("RAG Demo App")
    page = st.sidebar.radio("Select a page", ["Cost Database", "Canvas Data Processing"])

    if page == "Cost Database":
        cost_database_page()
    elif page == "Canvas Data Processing":
        canvas_data_page()

if __name__ == "__main__":
    main()
