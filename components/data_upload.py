import streamlit as st
import pandas as pd
from utils.data_processor import process_csv_data
from utils.openai_helper import get_embeddings, validate_openai_api_key
from utils.vector_store import VectorStore
import io
import logging
import os
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_csv() -> pd.DataFrame:
    '''Generate a sample CSV template with example legal document data'''
    return pd.DataFrame({
        'title': ['Contract Agreement', 'Privacy Policy', 'Terms of Service'],
        'content': [
            'This agreement is made between Party A and Party B...',
            'We collect and process personal data in accordance with...',
            'By using our services, you agree to the following terms...'
        ],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'department': ['Legal', 'Compliance', 'Legal'],
        'status': ['Active', 'Under Review', 'Active']
    })

def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate the structure of the uploaded CSV file."""
    if df.empty:
        return False, "The CSV file is empty."
    if len(df.columns) == 0:
        return False, "The CSV file must contain at least one column."
    if df.columns.duplicated().any():
        return False, "The CSV file contains duplicate column names."
    return True, ""

def render_upload_section():
    st.header("Upload Legal Data")
    
    # Instructions
    st.markdown("""
    ### Instructions
    1. Prepare your CSV file with legal documents data
    2. Each row should represent a separate document or section
    3. Include columns with text content (e.g., document title, content, clauses)
    4. Additional metadata columns (e.g., date, category, department) for advanced filtering
    """)
    
    # Sample template download
    st.subheader("Download Sample Template")
    sample_df = generate_sample_csv()
    buffer = io.BytesIO()
    sample_df.to_csv(buffer, index=False)
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=buffer.getvalue(),
        file_name="legal_documents_template.csv",
        mime="text/csv"
    )
    
    # File upload
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Check file size
            file_size = uploaded_file.size
            if file_size == 0:
                st.error("The uploaded file is empty. Please upload a valid CSV file.")
                logger.error("Empty file uploaded")
                return
            elif file_size > 100 * 1024 * 1024:  # 100MB limit
                st.error("File size exceeds 100MB limit. Please upload a smaller file.")
                logger.error(f"File size too large: {file_size / (1024*1024):.2f}MB")
                return
            
            logger.info(f"Processing file of size: {file_size / 1024:.2f}KB")
            
            # Reset file pointer and try to read CSV
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            logger.info(f"Successfully read CSV with shape: {df.shape}")
            
            # Preview the data
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Column selection
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Content Columns")
                st.markdown("Select columns containing the main text content for analysis.")
                selected_columns = st.multiselect(
                    "Select content columns",
                    options=df.columns.tolist(),
                    help="Choose columns containing document text, titles, or legal content"
                )
            
            with col2:
                st.markdown("### Metadata Columns")
                st.markdown("Select additional columns for filtering and categorization.")
                metadata_columns = st.multiselect(
                    "Select metadata columns",
                    options=[col for col in df.columns if col not in selected_columns],
                    help="Choose columns for filtering (e.g., date, category, department)"
                )
            
            # Process button with loading indicator
            if st.button("Process Data"):
                if not selected_columns:
                    st.error("Please select at least one column for processing.")
                    logger.error("No columns selected for processing")
                    return
                
                # Validate OpenAI API key before processing
                is_valid, message = validate_openai_api_key()
                if not is_valid:
                    st.error(f"Invalid OpenAI API key: {message}")
                    logger.error(f"OpenAI API key validation failed: {message}")
                    return
                
                with st.spinner("Processing data..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Process CSV data
                        status_text.text("Processing CSV data...")
                        progress_bar.progress(25)
                        logger.info("Starting data processing")
                        
                        df, texts, metadata_df = process_csv_data(uploaded_file, selected_columns, metadata_columns)
                        logger.info(f"Processed {len(texts)} documents")
                        
                        # Validate texts
                        if not texts or all(not text.strip() for text in texts):
                            st.error("Selected columns contain no text data. Please select columns with text content.")
                            logger.error("No text content found in selected columns")
                            return
                        
                        # Generate embeddings
                        status_text.text("Generating embeddings using OpenAI...")
                        progress_bar.progress(50)
                        logger.info("Starting embedding generation")
                        
                        try:
                            embeddings = get_embeddings(texts)
                            logger.info("Generated embeddings successfully")
                            
                            # Create vector store
                            status_text.text("Creating vector store...")
                            progress_bar.progress(75)
                            vector_store = VectorStore()
                            vector_store.create_index(embeddings, texts, metadata_df)
                            
                            # Save to session state
                            st.session_state.df = df
                            st.session_state.vector_store = vector_store
                            
                            # Final progress update
                            progress_bar.progress(100)
                            status_text.empty()
                            st.success("Data processed successfully! 🎉")
                            logger.info("Data processing completed successfully")
                            
                        except Exception as e:
                            error_msg = str(e)
                            st.error(f"Error generating embeddings: {error_msg}")
                            logger.error(f"Embedding generation failed: {error_msg}")
                            return
                            
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"An unexpected error occurred: {error_msg}")
                        logger.error(f"Unexpected error: {error_msg}", exc_info=True)
                        return
                        
        except Exception as e:
            error_msg = str(e)
            st.error(f"Error processing CSV file: {error_msg}")
            logger.error(f"CSV processing error: {error_msg}", exc_info=True)
