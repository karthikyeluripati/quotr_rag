import streamlit as st
import json
import faiss
import openai
import numpy as np
import pandas as pd
import uuid
import logging
import io
import traceback

# Set up logging
log_stream = io.StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=log_stream
)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
logger.info(f"OpenAI API key set: {openai.api_key[:5]}...")  # Log first 5 characters of API key

# Initialize session state
if 'cost_database' not in st.session_state:
    st.session_state.cost_database = []
if 'cost_embeddings' not in st.session_state:
    st.session_state.cost_embeddings = None
if 'cost_index' not in st.session_state:
    st.session_state.cost_index = None
if 'canvas_data' not in st.session_state:
    st.session_state.canvas_data = []
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
    except openai.error.AuthenticationError:
        logger.error("OpenAI API key is invalid.")
        st.error("OpenAI API key is invalid. Please check your API key in the Streamlit secrets.")
        return None
    except openai.error.RateLimitError:
        logger.error("OpenAI API rate limit exceeded.")
        st.error("OpenAI API rate limit exceeded. Please try again later.")
        return None
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error getting embedding: {str(e)}")
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
        index.add(np.array(embeddings).astype('float32'))
        logger.info(f"FAISS index created with {len(embeddings)} items")
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def update_cost_embeddings_and_index():
    try:
        embeddings = []
        for i, item in enumerate(st.session_state.cost_database):
            sentence = create_sentence(item)
            logger.info(f"Creating embedding for item {i+1}/{len(st.session_state.cost_database)}: {sentence[:100]}...")
            embedding = get_openai_embedding(sentence)
            if embedding is None:
                logger.error(f"Failed to get embedding for item {i+1}: {item.get('name', 'Unknown')}")
                continue
            embeddings.append(embedding)
            if (i + 1) % 10 == 0:  # Log progress every 10 items
                logger.info(f"Processed {i+1}/{len(st.session_state.cost_database)} items")
        
        if not embeddings:
            logger.error("No valid embeddings created")
            return None, None
        
        logger.info(f"Creating FAISS index with {len(embeddings)} embeddings")
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
        distances, indices = st.session_state.cost_index.search(np.array([query_embedding]).astype('float32'), 1)
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
            logger.info(f"Cost database uploaded successfully with {len(new_cost_database)} items")
            st.success(f"Cost database uploaded successfully! {len(new_cost_database)} items loaded.")
            
            # Automatically update embeddings and index
            with st.spinner("Updating embeddings and index..."):
                embeddings, index = update_cost_embeddings_and_index()
            if embeddings is not None and index is not None:
                logger.info(f"Embeddings and index updated successfully with {len(embeddings)} embeddings")
                st.success(f"Embeddings and index updated automatically! Created {len(embeddings)} embeddings.")
                
                # Display updated cost database
                st.subheader("Updated Cost Database")
                df = pd.DataFrame(st.session_state.cost_database)
                st.dataframe(df)
            else:
                logger.error("Failed to update embeddings and index")
                st.error("Failed to update embeddings and index. Please check the logs for details.")
        except json.JSONDecodeError:
            logger.error("Invalid JSON file uploaded")
            st.error("Error: Invalid JSON file. Please upload a valid JSON file.")
        except Exception as e:
            logger.error(f"Error uploading cost database: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error uploading cost database: {str(e)}")
            st.error(traceback.format_exc())

def canvas_data_page():
    st.title("Canvas Data Processing")

    if not st.session_state.cost_database or st.session_state.cost_embeddings is None or st.session_state.cost_index is None:
        logger.warning("Cost database or embeddings not found")
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
            logger.error(f"Error loading canvas data: {str(e)}")
            logger.error(traceback.format_exc())
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
                        logger.info(f"Match found for item: {item.get('Family', '')} - {item.get('Type', '')}")
                    else:
                        logger.warning(f"No match found for item: {item.get('Family', '')} - {item.get('Type', '')}")
                        st.warning(f"No match found for item: {item.get('Family', '')} - {item.get('Type', '')}")
                else:
                    logger.error(f"Failed to get embedding for item: {item.get('Family', '')} - {item.get('Type', '')}")
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
                logger.info(f"Processed {len(results)} canvas data items successfully")
            else:
                logger.warning("No matching results found")
                st.warning("No matching results found.")
    else:
        st.info("Please upload canvas data to process.")

def manual_best_match_page():
    st.title("Manual Best Match Check")

    if not st.session_state.cost_database or st.session_state.cost_embeddings is None or st.session_state.cost_index is None:
        logger.warning("Cost database or embeddings not found")
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
        logger.info(f"Manual input sentence: {sentence}")
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
                logger.info(f"Best match found for manual input: {best_match.get('name', '')}")
                
                if st.button("Add to Canvas Data"):
                    st.session_state.canvas_data.append(manual_input)
                    logger.info("Manual input added to Canvas Data")
                    st.success("Added to Canvas Data successfully!")
            else:
                logger.warning("No match found for the manually entered item")
                st.warning("No match found for the entered item.")
        else:
            logger.error("Failed to get embedding for the manually entered item")
            st.warning("Failed to get embedding for the entered item.")

def display_logs():
    st.sidebar.subheader("Logs")
    log_contents = log_stream.getvalue()
    if log_contents:
        st.sidebar.text_area("Log Output", log_contents, height=300)
    else:
        st.sidebar.text("No logs available.")

def main():
    st.sidebar.title("RAG Demo App")
    page = st.sidebar.radio("Select a page", [
        "Cost Database", 
        "Canvas Data Processing",
        "Manual Best Match Check"
    ])

    if page == "Cost Database":
        cost_database_page()
    elif page == "Canvas Data Processing":
        canvas_data_page()
    elif page == "Manual Best Match Check":
        manual_best_match_page()

    # Debug information
    st.sidebar.subheader("Debug Information")
    st.sidebar.write(f"Cost Database Items: {len(st.session_state.cost_database)}")
    st.sidebar.write(f"Cost Embeddings: {'Created' if st.session_state.cost_embeddings is not None else 'Not Created'}")
    st.sidebar.write(f"Cost Index: {'Created' if st.session_state.cost_index is not None else 'Not Created'}")
    st.sidebar.write(f"Canvas Data Items: {len(st.session_state.canvas_data)}")
    st.sidebar.write(f"Canvas Data Results: {len(st.session_state.canvas_data_results)}")

    # Display logs
    display_logs()

if __name__ == "__main__":
    main()