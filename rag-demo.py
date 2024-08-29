import streamlit as st
import json
import faiss
import openai
import numpy as np
import pandas as pd
import os
import pickle
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# File paths for storing data
COST_DB_FILE = "cost_database.json"
EMBEDDINGS_FILE = "embeddings.pkl"
INDEX_FILE = "faiss_index.pkl"

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

def save_embeddings_and_index(embeddings, index):
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    faiss.write_index(index, INDEX_FILE)
    logger.info(f"Embeddings and index saved with {len(embeddings)} items")

def load_embeddings_and_index():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings = pickle.load(f)
        index = faiss.read_index(INDEX_FILE)
        logger.info(f"Embeddings and index loaded with {len(embeddings)} items")
        return embeddings, index
    logger.warning("Embeddings or index file not found")
    return None, None

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

def create_faiss_index(embeddings):
    if not embeddings or len(embeddings) == 0:
        logger.error("No embeddings provided to create FAISS index")
        return None
    try:
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        logger.info(f"FAISS index created with {len(embeddings)} items")
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        return None

def find_best_match(query_embedding, index, cost_database):
    if query_embedding is None or index is None:
        logger.error("Invalid query embedding or index for finding best match")
        return None, None
    try:
        distances, indices = index.search(np.array([query_embedding]), 1)
        best_match_index = indices[0][0]
        return cost_database[best_match_index], distances[0][0]
    except Exception as e:
        logger.error(f"Error finding best match: {str(e)}")
        return None, None

def create_sentence(item, is_cost_db=True):
    if is_cost_db:
        fields = ["csiSection", "csiDivisionName", "csiTitle", "nahbCodeDescription", "name", "nahbCategory", "nahbFamily", "nahbType", "description"]
    else:
        fields = ["Family", "Type", "Height", "Width"]
    
    sentence = " ".join(str(item.get(field, "")) for field in fields if field in item)
    logger.info(f"Created sentence: {sentence[:100]}...")  # Log first 100 characters
    return sentence

def update_embeddings_and_index(cost_database):
    try:
        embeddings = [get_openai_embedding(create_sentence(item)) for item in cost_database]
        if not embeddings or len(embeddings) == 0:
            logger.error("No valid embeddings created")
            return None, None
        index = create_faiss_index(embeddings)
        if index is not None:
            save_embeddings_and_index(embeddings, index)
            logger.info("Embeddings and index updated successfully")
            return embeddings, index
        else:
            logger.error("Failed to create FAISS index")
            return None, None
    except Exception as e:
        logger.error(f"Error updating embeddings and index: {str(e)}")
        return None, None

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
        try:
            new_cost_database = json.load(uploaded_file)
            for item in new_cost_database:
                if 'id' not in item:
                    item['id'] = str(uuid.uuid4())
            save_cost_database(new_cost_database)
            st.success("Cost database uploaded successfully!")
            
            # Automatically update embeddings and index
            embeddings, index = update_embeddings_and_index(new_cost_database)
            if embeddings is not None and index is not None:
                st.success("Embeddings and index updated automatically!")
            else:
                st.error("Failed to update embeddings and index. Please check the logs.")
        except Exception as e:
            st.error(f"Error uploading cost database: {str(e)}")

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
        
        # Automatically update embeddings and index
        embeddings, index = update_embeddings_and_index(cost_database)
        if embeddings is not None and index is not None:
            st.success("Embeddings and index updated automatically!")
        else:
            st.error("Failed to update embeddings and index. Please check the logs.")

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
                    st.success("Item updated successfully!")
                    
                    # Automatically update embeddings and index
                    embeddings, index = update_embeddings_and_index(cost_database)
                    if embeddings is not None and index is not None:
                        st.success("Embeddings and index updated automatically!")
                    else:
                        st.error("Failed to update embeddings and index. Please check the logs.")
            
            with col2:
                if st.button("Delete Item"):
                    cost_database.remove(item_to_edit)
                    save_cost_database(cost_database)
                    st.success("Item deleted successfully!")
                    
                    # Automatically update embeddings and index
                    embeddings, index = update_embeddings_and_index(cost_database)
                    if embeddings is not None and index is not None:
                        st.success("Embeddings and index updated automatically!")
                    else:
                        st.error("Failed to update embeddings and index. Please check the logs.")
        else:
            st.error("Item not found.")

def canvas_data_page():
    st.title("Canvas Data Processing")

    cost_database = load_cost_database()
    embeddings, index = load_embeddings_and_index()

    if not cost_database or embeddings is None or index is None:
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
        if uploaded_file:
            try:
                canvas_data_json = json.load(uploaded_file)
                canvas_data = canvas_data_json[0].get("Automated Door Schedule", [])
                logger.info(f"Loaded canvas data with {len(canvas_data)} items")
            except Exception as e:
                st.error(f"Error loading canvas data: {str(e)}")
                return
        else:
            canvas_data = [manual_input]
            logger.info("Processing manual input")

        results = []
        for item in canvas_data:
            sentence = create_sentence(item, is_cost_db=False)
            query_embedding = get_openai_embedding(sentence)
            best_match, distance = find_best_match(query_embedding, index, cost_database)
            if best_match is not None:
                results.append({
                    "Canvas Item": item.get('Family', '') + ' - ' + item.get('Type', ''),
                    "Matched Item": best_match.get('name', ''),
                    "Matched Description": best_match.get('description', ''),
                    "Matched Material Rate (USD)": best_match.get('materialRateUsd', ''),
                    "Matched Burdened Labor Rate (USD)": best_match.get('burdenedLaborRateUsd', ''),
                    "Similarity Score": 1 / (1 + distance)
                })
            else:
                st.warning(f"No match found for item: {item.get('Family', '')} - {item.get('Type', '')}")

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
            st.warning("No matching results found.")

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