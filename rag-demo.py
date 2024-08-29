import streamlit as st
import json
import faiss
import openai
import numpy as np
import pandas as pd
import uuid
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize session state
if 'cost_database' not in st.session_state:
    st.session_state.cost_database = []
if 'cost_embeddings' not in st.session_state:
    st.session_state.cost_embeddings = None
if 'cost_index' not in st.session_state:
    st.session_state.cost_index = None
if 'canvas_data' not in st.session_state:
    st.session_state.canvas_data = None
if 'canvas_embeddings' not in st.session_state:
    st.session_state.canvas_embeddings = None
if 'canvas_data_results' not in st.session_state:
    st.session_state.canvas_data_results = []

# Helper functions
def get_openai_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_sentence(item, is_cost_db=True):
    if is_cost_db:
        fields = ["csiSection", "csiDivisionName", "csiTitle", "nahbCodeDescription", "name", "nahbCategory", "nahbFamily", "nahbType", "description"]
    else:
        fields = ["Family", "Type", "Height", "Width"]
    
    sentence = " ".join(str(item.get(field, "")) for field in fields if field in item)
    logger.info(f"Created sentence: {sentence[:100]}...")  # Log first 100 characters
    return sentence

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
        logger.error(traceback.format_exc())
        return None

def update_cost_embeddings_and_index():
    try:
        embeddings = []
        for item in st.session_state.cost_database:
            sentence = create_sentence(item)
            embedding = get_openai_embedding(sentence)
            if embedding is None:
                logger.error(f"Failed to get embedding for item: {item.get('name', 'Unknown')}")
                continue
            embeddings.append(embedding)
        
        if not embeddings:
            logger.error("No valid embeddings created")
            return None, None
        
        index = create_faiss_index(embeddings)
        if index is not None:
            st.session_state.cost_embeddings = embeddings
            st.session_state.cost_index = index
            logger.info("Cost embeddings and index updated successfully")
            return embeddings, index
        else:
            logger.error("Failed to create FAISS index")
            return None, None
    except Exception as e:
        logger.error(f"Error updating cost embeddings and index: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def find_best_match(query_embedding):
    if query_embedding is None or st.session_state.cost_index is None:
        logger.error("Invalid query embedding or index for finding best match")
        return None, None
    try:
        distances, indices = st.session_state.cost_index.search(np.array([query_embedding]), 1)
        best_match_index = indices[0][0]
        return st.session_state.cost_database[best_match_index], distances[0][0]
    except Exception as e:
        logger.error(f"Error finding best match: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

# Page functions
def cost_database_page():
    st.title("Cost Database Management")

    st.subheader("Current Cost Database")
    if st.session_state.cost_database:
        df = pd.DataFrame(st.session_state.cost_database)
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
            st.session_state.cost_database = new_cost_database
            st.success("Cost database uploaded successfully!")
            
            # Automatically update embeddings and index
            with st.spinner("Updating embeddings and index..."):
                embeddings, index = update_cost_embeddings_and_index()
            if embeddings is not None and index is not None:
                st.success(f"Embeddings and index updated automatically! Created {len(embeddings)} embeddings.")
            else:
                st.error("Failed to update embeddings and index. Please check the logs for details.")
                st.error("You may need to check your OpenAI API key or network connection.")
        except json.JSONDecodeError:
            st.error("Error: Invalid JSON file. Please upload a valid JSON file.")
        except Exception as e:
            st.error(f"Error uploading cost database: {str(e)}")
            st.error(traceback.format_exc())

    # Display current embeddings and index info
    if st.session_state.cost_embeddings is not None and st.session_state.cost_index is not None:
        st.subheader("Current Embeddings and Index Information")
        st.write(f"Number of embeddings: {len(st.session_state.cost_embeddings)}")
        st.write(f"Embedding dimension: {len(st.session_state.cost_embeddings[0])}")
        st.write(f"Index size: {st.session_state.cost_index.ntotal}")
    else:
        st.warning("No embeddings or index created yet. Please upload a cost database.")

def add_new_item_page():
    st.title("Add New Item to Cost Database")

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
        st.session_state.cost_database.append(new_item)
        st.success("Item added successfully!")
        
        # Automatically update embeddings and index
        with st.spinner("Updating embeddings and index..."):
            embeddings, index = update_cost_embeddings_and_index()
        if embeddings is not None and index is not None:
            st.success("Embeddings and index updated automatically!")
        else:
            st.error("Failed to update embeddings and index. Please check the logs for details.")
            st.error("You may need to check your OpenAI API key or network connection.")

def edit_delete_page():
    st.title("Edit/Delete Items")

    search_term = st.text_input("Enter item ID or name to edit/delete")
    if search_term:
        item_to_edit = next((item for item in st.session_state.cost_database if item['id'] == search_term or item['name'] == search_term), None)
        if item_to_edit:
            st.write(f"Editing item: {item_to_edit['name']} (ID: {item_to_edit['id']})")
            edit_item = item_to_edit.copy()
            for key in edit_item:
                if key != 'id':
                    edit_item[key] = st.text_input(f"Edit {key}", edit_item[key])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update Item"):
                    index = st.session_state.cost_database.index(item_to_edit)
                    st.session_state.cost_database[index] = edit_item
                    st.success("Item updated successfully!")
                    
                    # Automatically update embeddings and index
                    with st.spinner("Updating embeddings and index..."):
                        embeddings, index = update_cost_embeddings_and_index()
                    if embeddings is not None and index is not None:
                        st.success("Embeddings and index updated automatically!")
                    else:
                        st.error("Failed to update embeddings and index. Please check the logs for details.")
                        st.error("You may need to check your OpenAI API key or network connection.")
            
            with col2:
                if st.button("Delete Item"):
                    st.session_state.cost_database.remove(item_to_edit)
                    st.success("Item deleted successfully!")
                    
                    # Automatically update embeddings and index
                    with st.spinner("Updating embeddings and index..."):
                        embeddings, index = update_cost_embeddings_and_index()
                    if embeddings is not None and index is not None:
                        st.success("Embeddings and index updated automatically!")
                    else:
                        st.error("Failed to update embeddings and index. Please check the logs for details.")
                        st.error("You may need to check your OpenAI API key or network connection.")
        else:
            st.error("Item not found.")

def canvas_data_page():
    st.title("Canvas Data Processing")

    if not st.session_state.cost_database or st.session_state.cost_embeddings is None or st.session_state.cost_index is None:
        st.error("Please upload and process the cost database first.")
        return

    uploaded_file = st.file_uploader("Upload Canvas Data (JSON)", type="json")
    
    if uploaded_file is not None:
        try:
            canvas_data_json = json.load(uploaded_file)
            st.session_state.canvas_data = canvas_data_json[0].get("Automated Door Schedule", [])
            logger.info(f"Loaded canvas data with {len(st.session_state.canvas_data)} items")
            st.success(f"Canvas data loaded with {len(st.session_state.canvas_data)} items")
        except Exception as e:
            st.error(f"Error loading canvas data: {str(e)}")
            st.error(traceback.format_exc())
            return

    if st.session_state.canvas_data:
        if st.button("Process Canvas Data"):
            results = []
            for item in st.session_state.canvas_data:
                sentence = create_sentence(item, is_cost_db=False)
                query_embedding = get_openai_embedding(sentence)
                if query_embedding is not None:
                    best_match, distance = find_best_match(query_embedding)
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
                else:
                    st.warning(f"Failed to get embedding for item: {item.get('Family', '')} - {item.get('Type', '')}")

            if results:
                st.session_state.canvas_data_results = results
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
    else:
        st.info("Please upload canvas data to process.")

    # Edit/Delete Canvas Data Results
    if st.session_state.canvas_data_results:
        st.subheader("Edit or Delete Canvas Data Results")
        index_to_edit = st.number_input("Enter the index of the entry to edit/delete", min_value=0, max_value=len(st.session_state.canvas_data_results)-1, value=0)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Edit Entry"):
                if 0 <= index_to_edit < len(st.session_state.canvas_data_results):
                    entry = st.session_state.canvas_data_results[index_to_edit]
                    edited_entry = {}
                    for key, value in entry.items():
                        edited_entry[key] = st.text_input(f"Edit {key}", value)
                    
                    if st.button("Save Changes"):
                        st.session_state.canvas_data_results[index_to_edit] = edited_entry
                        st.success("Entry updated successfully!")
                else:
                    st.error("Invalid index.")

        with col2:
            if st.button("Delete Entry"):
                if 0 <= index_to_edit < len(st.session_state.canvas_data_results):
                    del st.session_state.canvas_data_results[index_to_edit]
                    st.success("Entry deleted successfully!")
                else:
                    st.error("Invalid index.")

        if st.button("Download Updated Canvas Data Results"):
            df = pd.DataFrame(st.session_state.canvas_data_results)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Updated CSV",
                data=csv,
                file_name="updated_canvas_data_results.csv",
                mime="text/csv",
            )

def manual_best_match_page():
    st.title("Manual Best Match Check")

    if not st.session_state.cost_database or st.session_state.cost_embeddings is None or st.session_state.cost_index is None:
        st.error("Please upload and process the cost database first.")
        return

    st.subheader("Enter Canvas Data")
    manual_input = {}
    manual_input['Family'] = st.text_input("Family")
    manual_input['Type'] = st.text_input("Type")
    manual_input['Height'] = st.text_input("Height")
    manual_input['Width'] = st.text_input("Width")
    manual_input['Assembly Code'] = st.text_input("Assembly Code")
    manual_input['Level'] = st.text_input("Level")
    manual_input['From Room: Number'] = st.text_input("From Room: Number")

    if st.button("Find Best Match"):
        sentence = create_sentence(manual_input, is_cost_db=False)
        query_embedding = get_openai_embedding(sentence)
        if query_embedding is not None:
            best_match, distance = find_best_match(query_embedding)
            if best_match is not None:
                st.subheader("Best Match Found")
                match_result = {
                    "Canvas Item": manual_input['Family'] + ' - ' + manual_input['Type'],
                    "Matched Item": best_match.get('name', ''),
                    "Matched Description": best_match.get('description', ''),
                    "Matched Material Rate (USD)": best_match.get('materialRateUsd', ''),
                    "Matched Burdened Labor Rate (USD)": best_match.get('burdenedLaborRateUsd', ''),
                    "Similarity Score": 1 / (1 + distance)
                }
                st.json(match_result)
                
                if st.button("Add to Canvas Data Results"):
                    if 'canvas_data_results' not in st.session_state:
                        st.session_state.canvas_data_results = []
                    st.session_state.canvas_data_results.append(match_result)
                    st.success("Added to Canvas Data Results successfully!")
            else:
                st.warning("No match found for the entered item.")
        else:
            st.warning("Failed to get embedding for the entered item.")

def main():
    st.sidebar.title("RAG Demo App")
    page = st.sidebar.radio("Select a page", [
        "Cost Database", 
        "Add New Item", 
        "Edit/Delete Items", 
        "Canvas Data Processing",
        "Manual Best Match Check"
    ])

    if page == "Cost Database":
        cost_database_page()
    elif page == "Add New Item":
        add_new_item_page()
    elif page == "Edit/Delete Items":
        edit_delete_page()
    elif page == "Canvas Data Processing":
        canvas_data_page()
    elif page == "Manual Best Match Check":
        manual_best_match_page()

if __name__ == "__main__":
    main()