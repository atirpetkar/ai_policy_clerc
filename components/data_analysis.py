import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.data_processor import generate_basic_statistics
from utils.text_analysis import (
    generate_word_cloud,
    extract_key_phrases,
    analyze_text_length,
    perform_topic_modeling,
    calculate_similarity_matrix
)
from utils.export_utils import export_analysis_results

def render_analysis_section():
    st.header("Data Analysis")
    
    if st.session_state.df is None:
        st.warning("Please upload and process data first.")
        return
    
    df = st.session_state.df
    
    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["Basic Analysis", "Text Analysis"])
    
    with tab1:
        render_basic_analysis(df)
    
    with tab2:
        render_text_analysis(df)

def render_basic_analysis(df):
    # Display basic statistics
    stats = generate_basic_statistics(df)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Statistics")
        st.write(f"Total Rows: {stats['total_rows']}")
        st.write(f"Total Columns: {stats['total_columns']}")
    
    with col2:
        st.subheader("Column Types")
        for col, dtype in stats['column_types'].items():
            st.write(f"{col}: {dtype}")
    
    # Add export button for basic statistics
    export_analysis_results(
        pd.DataFrame([stats]),
        "Basic Statistics",
        format='json'
    )
    
    # Missing values visualization
    st.subheader("Missing Values Analysis")
    missing_df = pd.DataFrame.from_dict(stats['missing_values'], orient='index', columns=['count'])
    fig = px.bar(missing_df, x=missing_df.index, y='count', title='Missing Values by Column')
    st.plotly_chart(fig)
    
    # Export missing values data
    export_analysis_results(missing_df, "Missing Values Analysis")
    
    # Column analysis
    st.subheader("Column Analysis")
    selected_column = st.selectbox("Select column for analysis", df.columns)
    
    if df[selected_column].dtype in ['int64', 'float64']:
        fig = px.histogram(df, x=selected_column, title=f'Distribution of {selected_column}')
        st.plotly_chart(fig)
        
        # Export numerical analysis
        analysis_df = pd.DataFrame({
            'value': df[selected_column].values
        })
        export_analysis_results(analysis_df, f"{selected_column} Distribution")
    else:
        value_counts = df[selected_column].value_counts().head(10)
        fig = px.pie(values=value_counts.values, names=value_counts.index, title=f'Top 10 Values in {selected_column}')
        st.plotly_chart(fig)
        
        # Export categorical analysis
        export_analysis_results(
            pd.DataFrame(value_counts).reset_index(),
            f"{selected_column} Distribution"
        )

def render_text_analysis(df):
    st.subheader("Text Analysis")
    
    # Get text columns
    text_columns = df.select_dtypes(include=['object']).columns
    selected_text_column = st.selectbox("Select text column for analysis", text_columns)
    
    if selected_text_column:
        texts = df[selected_text_column].fillna('').tolist()
        
        # Word Cloud
        st.subheader("Word Cloud")
        try:
            wordcloud_array = generate_word_cloud(texts)
            st.image(wordcloud_array, use_column_width=True)
        except Exception as e:
            st.error(f"Error generating word cloud: {str(e)}")
        
        # Key Phrases
        st.subheader("Top Key Phrases")
        try:
            key_phrases = extract_key_phrases(texts)
            phrases_df = pd.DataFrame(key_phrases, columns=['Phrase', 'Importance'])
            fig = px.bar(phrases_df, x='Phrase', y='Importance', title='Top Key Phrases')
            st.plotly_chart(fig)
            
            # Export key phrases
            export_analysis_results(phrases_df, "Key Phrases Analysis")
            
        except Exception as e:
            st.error(f"Error extracting key phrases: {str(e)}")
        
        # Text Length Distribution
        st.subheader("Text Length Distribution")
        try:
            lengths = analyze_text_length(texts)
            fig = px.histogram(x=lengths, title='Distribution of Text Lengths (words)')
            st.plotly_chart(fig)
            
            # Export text length data
            export_analysis_results(
                pd.DataFrame({'text_length': lengths}),
                "Text Length Distribution"
            )
            
        except Exception as e:
            st.error(f"Error analyzing text length: {str(e)}")
        
        # Topic Modeling
        st.subheader("Topic Analysis")
        try:
            n_topics = st.slider("Number of topics", min_value=2, max_value=10, value=5)
            topics, topic_distribution = perform_topic_modeling(texts, n_topics=n_topics)
            
            # Display topics
            for topic_name, topic_words in topics:
                st.write(f"**{topic_name}:** {topic_words}")
            
            # Topic distribution heatmap
            topic_dist_df = pd.DataFrame(
                topic_distribution,
                columns=[f"Topic {i+1}" for i in range(n_topics)]
            )
            fig = px.imshow(
                topic_dist_df.corr(),
                title='Topic Correlation Heatmap',
                labels=dict(color="Correlation")
            )
            st.plotly_chart(fig)
            
            # Export topic analysis results
            export_analysis_results(
                {'topics': topics, 'distribution': topic_distribution.tolist()},
                "Topic Analysis",
                format='json'
            )
            
        except Exception as e:
            st.error(f"Error in topic modeling: {str(e)}")
        
        # Document Similarity
        st.subheader("Document Similarity Analysis")
        try:
            similarity_matrix = calculate_similarity_matrix(texts)
            fig = px.imshow(
                similarity_matrix,
                title='Document Similarity Heatmap',
                labels=dict(color="Similarity Score")
            )
            st.plotly_chart(fig)
            
            # Export similarity matrix
            export_analysis_results(
                pd.DataFrame(similarity_matrix),
                "Document Similarity Matrix"
            )
            
        except Exception as e:
            st.error(f"Error calculating document similarity: {str(e)}")
