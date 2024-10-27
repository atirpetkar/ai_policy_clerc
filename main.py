import streamlit as st
from components.data_upload import render_upload_section
from components.data_analysis import render_analysis_section
from components.chatbot import render_chatbot_section

st.set_page_config(
    page_title="Sue-per GPT",
    page_icon="⚖️",
    layout="wide",
)

def main():
    st.title("🤖 Sue-per GPT")
    st.markdown("*Bringing Law to Your Screen Faster Than You Can Say 'Objection!'*")
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "📊 Analysis", "💬 Chat"])
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_analysis_section()
    
    with tab3:
        render_chatbot_section()

if __name__ == "__main__":
    main()
