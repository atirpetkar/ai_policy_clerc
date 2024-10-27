import streamlit as st
from utils.openai_helper import get_embeddings, get_chat_response, validate_openai_api_key
from utils.text_analysis import advanced_search
from utils.export_utils import export_chat_history
import os

def validate_and_update_api_key(key: str) -> None:
    """Validate and update OpenAI API key with immediate feedback."""
    if not key:
        return
    
    os.environ["OPENAI_API_KEY"] = key
    is_valid, message = validate_openai_api_key()
    
    if is_valid:
        st.sidebar.success("OpenAI API key validated and updated successfully! ✅")
    else:
        st.sidebar.error(f"❌ {message}")
        os.environ.pop("OPENAI_API_KEY", None)

def render_chatbot_section():
    st.header("Sue-per GPT Chat")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Advanced Search and Filtering Options
    with st.sidebar:
        # API Key Configuration
        st.markdown("# 🔑 API Key Configuration")
        st.markdown("""
        Enter your OpenAI API key to use the chatbot functionality. 
        The key is securely stored and validated immediately.
        """)
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key for GPT-4 access"
        )
        
        # Validate OpenAI key when changed
        if openai_key != os.environ.get("OPENAI_API_KEY", ""):
            validate_and_update_api_key(openai_key)
            
        st.markdown("---")  # Separator
            
        # Advanced search options
        st.header("Advanced Search Options")
        
        # Text Length Filter
        st.subheader("Text Length Filter")
        min_length = st.slider(
            "Minimum Text Length (words)",
            min_value=0,
            max_value=500,
            value=0,
            help="Filter documents by minimum word count"
        )
        
        # Metadata filters
        st.header("Metadata Filters")
        metadata_fields = st.session_state.vector_store.get_metadata_fields() if st.session_state.vector_store else []
        active_filters = {}
        
        for field in metadata_fields:
            unique_values = st.session_state.vector_store.get_unique_values(field)
            if unique_values:
                selected_values = st.multiselect(
                    f"Filter by {field}",
                    options=unique_values,
                    help=f"Select values to filter {field}"
                )
                if selected_values:
                    active_filters[field] = selected_values
        
        # Clear filters button
        if st.button("Clear All Filters"):
            st.session_state.messages = []
            st.rerun()
        
        # Export options
        st.header("Export Options")
        export_format = st.radio("Choose export format:", ['JSON', 'CSV'], horizontal=True)
        if st.button("Export Chat History"):
            export_chat_history(st.session_state.messages, format=export_format.lower())
    
    # Main chat area
    if st.session_state.vector_store is None:
        st.info("👆 Please configure your OpenAI API key in the sidebar.")
        st.warning("Please upload and process data to start chatting.")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input handling
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API key in the sidebar to start chatting.")
        return
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your legal documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get embedding for the query
                    query_embedding = get_embeddings([prompt])[0]
                    
                    # Combine filters
                    filters = {
                        'min_length': min_length if min_length > 0 else None,
                        **active_filters
                    }
                    
                    # Search relevant documents with both vector and text-based search
                    vector_texts, _, metadata = st.session_state.vector_store.search(
                        query_embedding,
                        k=2,  # Reduced from 3 to 2 documents
                        filters=active_filters if active_filters else None
                    )
                    
                    # Perform advanced search with additional filters
                    advanced_matches = advanced_search(
                        vector_texts,
                        prompt,
                        filters={k: v for k, v in filters.items() if v}
                    )
                    
                    # Get filtered context
                    if advanced_matches:
                        relevant_texts = [text for _, _, text in advanced_matches[:2]]  # Limited to 2 documents
                        context = "\n\n".join(relevant_texts)
                    else:
                        context = "\n\n".join(vector_texts)
                    
                    # Add metadata context if available (with length limit)
                    if not metadata.empty:
                        metadata_str = metadata.to_string()
                        if len(metadata_str) > 500:  # Limit metadata context
                            metadata_str = metadata_str[:500] + "..."
                        context += "\n\nRelevant metadata:\n" + metadata_str
                    
                    # Get response from OpenAI
                    response = get_chat_response(prompt, context)
                    
                    # Truncate long responses if needed
                    if len(response) > 2000:
                        response = response[:2000] + "...\n[Response truncated for brevity]"
                    
                    # Display response
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Display search context (optional)
                    with st.expander("View Search Context"):
                        st.write("Relevant Documents:")
                        for idx, (score, _, text) in enumerate(advanced_matches[:2], 1):  # Limited to 2 documents
                            st.write(f"{idx}. Relevance Score: {score:.2f}")
                            st.text(text[:200] + "...")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
