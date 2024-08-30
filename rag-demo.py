import streamlit as st
import json
import openai
import numpy as np
import pandas as pd
import os
import uuid
import logging
import time
from pinecone import Pinecone, PodSpec

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Pinecone initialization
pc = Pinecone(
    api_key=st.secrets["PINECONE_API_KEY"],
    environment=st.secrets["PINECONE_ENVIRONMENT"]
)

# Pinecone index name
INDEX_NAME = "cost-database-index"

# File paths for storing data
COST_DB_FILE = "cost_database.json"

# Connect to the existing Pinecone index
index = pc.Index(INDEX_NAME)

# Verify that the index exists
try:
    index_description = pc.describe_index(INDEX_NAME)
    st.success(f"Successfully connected to index '{INDEX_NAME}'")
    st.info(f"Index stats: {index_description.status}")
except Exception as e:
    st.error(f"Error connecting to index '{INDEX_NAME}': {str(e)}")
    st.error("Please check your Pinecone API key, environment, and index name.")
    st.stop()

# Helper functions
def save_cost_database(data):
    with open(COST_DB_FILE, 'w') as f:
        json.dump(data, f)
    logger.info(f"Cost database saved with {len(data)} items")

def load_cost_database():
    if os.path.exists(COST_DB_FILE):
        with open(COST_DB_FILE, 'r') as f:
            data = json.load(f)
        logger.info(f"Cost database loaded with {len(data)} items")
        return data
    logger.warning("Cost database file not found")
    return []

def get_openai_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return None

def update_pinecone_index(cost_database):
    to_upsert = []
    for item in cost_database:
        sentence = create_sentence(item)
        embedding = get_openai_embedding(sentence)
        if embedding:
            to_upsert.append((item['id'], embedding, item))
    
    # Upsert to Pinecone
    index.upsert(vectors=to_upsert)
    logger.info(f"Pinecone index updated with {len(to_upsert)} items")

def find_best_match(query_embedding, cost_database):
    query_response = index.query(vector=query_embedding, top_k=1, include_metadata=True)
    if query_response.matches:
        best_match = query_response.matches[0].metadata
        distance = query_response.matches[0].score
        return best_match, 1 - distance  # Convert similarity to distance
    return None, None

def create_sentence(item, is_cost_db=True):
    if is_cost_db:
        fields = ["csiSection", "csiDivisionName", "csiTitle", "nahbCodeDescription", "name", "nahbCategory", "nahbFamily", "nahbType", "description"]
        sentence = " ".join(str(item.get(field, "")).strip() for field in fields if field in item)
    else:
        exclude_fields = ["Assembly Code", "Level", "Room: Number", "From Room: Number"]
        sentence = " ".join(str(value).strip() for key, value in item.items() if value and key not in exclude_fields)
    
    logger.info(f"Created sentence: {sentence[:100]}...")  # Log first 100 characters
    return sentence

# Page functions
def cost_database_page():
    st.title("Cost Database Management")

    cost_database = load_cost_database()

    st.subheader("Current Cost Database")
    if cost_database:
        df = pd.DataFrame(cost_database)
        st.dataframe(df)
    else:
        st.write("No items in the database yet.")

    uploaded_file = st.file_uploader("Upload Cost Database (JSON)", type="json")
    if uploaded_file is not None:
        cost_database = json.load(uploaded_file)
        for item in cost_database:
            if 'id' not in item:
                item['id'] = str(uuid.uuid4())
        save_cost_database(cost_database)
        st.success("Cost database uploaded successfully!")

    if st.button("Update Embeddings"):
        with st.spinner("Updating embeddings..."):
            update_pinecone_index(cost_database)
        st.success("Embeddings updated successfully!")

def add_new_item_page():
    st.title("Add New Item to Cost Database")

    cost_database = load_cost_database()

    new_item = {}
    new_item['id'] = str(uuid.uuid4())
    new_item['name'] = st.text_input("Item Name")
    new_item['csiSection'] = st.text_input("CSI Section")
    new_item['csiDivisionName'] = st.text_input("CSI Division Name")
    new_item['csiTitle'] = st.text_input("CSI Title")
    new_item['nahbCodeDescription'] = st.text_input("NAHB Code Description")
    new_item['nahbCategory'] = st.text_input("NAHB Category")
    new_item['nahbFamily'] = st.text_input("NAHB Family")
    new_item['nahbType'] = st.text_input("NAHB Type")
    new_item['description'] = st.text_area("Description")
    new_item['materialRateUsd'] = st.number_input("Material Rate (USD)", min_value=0.0, format="%.2f")
    new_item['burdenedLaborRateUsd'] = st.number_input("Burdened Labor Rate (USD)", min_value=0.0, format="%.2f")

    if st.button("Add Item"):
        cost_database.append(new_item)
        save_cost_database(cost_database)
        st.success("Item added successfully!")
        
        # Update Pinecone index
        sentence = create_sentence(new_item)
        embedding = get_openai_embedding(sentence)
        if embedding:
            index.upsert(vectors=[(new_item['id'], embedding, new_item)])
            st.success("Embeddings updated successfully!")
        else:
            st.warning("Failed to update embeddings. Please update embeddings from the Cost Database page.")

def edit_delete_page():
    st.title("Edit/Delete Items")

    cost_database = load_cost_database()

    search_term = st.text_input("Enter item ID or name to edit/delete")
    if search_term:
        item_to_edit = next((item for item in cost_database if item['id'] == search_term or item['name'] == search_term), None)
        if item_to_edit:
            st.write(f"Editing item: {item_to_edit['name']} (ID: {item_to_edit['id']})")
            edit_item = item_to_edit.copy()
            for key in edit_item:
                if key != 'id':
                    edit_item[key] = st.text_input(f"Edit {key}", edit_item[key])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update Item"):
                    index = cost_database.index(item_to_edit)
                    cost_database[index] = edit_item
                    save_cost_database(cost_database)
                    # Update Pinecone index
                    sentence = create_sentence(edit_item)
                    embedding = get_openai_embedding(sentence)
                    if embedding:
                        index.upsert(vectors=[(edit_item['id'], embedding, edit_item)])
                    st.success("Item updated successfully!")
            with col2:
                if st.button("Delete Item"):
                    cost_database.remove(item_to_edit)
                    save_cost_database(cost_database)
                    # Delete from Pinecone index
                    try:
                        st.success("Item deleted successfully!")
                        index.delete(ids=[item_to_edit['id']])
                    except Exception as e:
                        st.error(f"Error deleting item from Pinecone index: {str(e)}")
                        st.warning("Item was removed from local database but may still exist in Pinecone index.")
        else:
            st.error("Item not found.")

def canvas_data_page():
    st.title("Canvas Data Processing")

    cost_database = load_cost_database()

    if not cost_database:
        st.error("Please upload and process the cost database first.")
        return

    uploaded_file = st.file_uploader("Upload Canvas Data (JSON)", type="json")
    
    st.subheader("Or Enter Canvas Data Manually")
    manual_input = {}
    manual_input['Family'] = st.text_input("Family")
    manual_input['Type'] = st.text_input("Type")
    manual_input['Height'] = st.text_input("Height")
    manual_input['Width'] = st.text_input("Width")
    manual_input['Assembly Code'] = st.text_input("Assembly Code")
    manual_input['Level'] = st.text_input("Level")
    manual_input['From Room: Number'] = st.text_input("From Room: Number")

    process_button = st.button("Process Data")

    if uploaded_file is not None or process_button:
        results = []
        if uploaded_file:
            canvas_data_json = json.load(uploaded_file)
            for schedule_name, canvas_data in canvas_data_json.items():
                logger.info(f"Processing {schedule_name} with {len(canvas_data)} items")
                results.extend(process_canvas_data(canvas_data, cost_database, schedule_name))
        else:
            canvas_data = [manual_input]
            logger.info("Processing manual input")
            results = process_canvas_data(canvas_data, cost_database, "Manual Input")

        if results:
            st.subheader("Matching Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="canvas_matching_results.csv",
                mime="text/csv",
            )
        else:
            st.error("No matching results found. Please check your input data and try again.")

def process_canvas_data(canvas_data, cost_database, schedule_name):
    results = []
    for item in canvas_data:
        sentence = create_sentence(item, is_cost_db=False)
        query_embedding = get_openai_embedding(sentence)
        if query_embedding:
            best_match, distance = find_best_match(query_embedding, cost_database)
            if best_match:
                results.append({
                    "Schedule": schedule_name,
                    "Canvas Item": item.get('Family', '') + ' - ' + item.get('Type', ''),
                    "Matched Item": best_match.get('name', ''),
                    "Matched Description": best_match.get('description', ''),
                    "Matched Material Rate (USD)": best_match.get('materialRateUsd', ''),
                    "Matched Burdened Labor Rate (USD)": best_match.get('burdenedLaborRateUsd', ''),
                    "Similarity Score": 1 / (1 + distance)
                })
            else:
                st.warning(f"No match found for item: {item.get('Family', '')} - {item.get('Type', '')} in {schedule_name}")
        else:
            st.warning(f"Failed to generate embedding for item: {item.get('Family', '')} - {item.get('Type', '')} in {schedule_name}")
    return results
def main():
    st.sidebar.title("RAG Demo App")
    page = st.sidebar.radio("Select a page", ["Cost Database", "Add New Item", "Edit/Delete Items", "Canvas Data Processing"])

    if page == "Cost Database":
        cost_database_page()
    elif page == "Add New Item":
        add_new_item_page()
    elif page == "Edit/Delete Items":
        edit_delete_page()
    elif page == "Canvas Data Processing":
        canvas_data_page()

if __name__ == "__main__":
    main()