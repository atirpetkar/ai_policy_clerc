import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import time
import re

def preprocess_text(text):
    """Basic text preprocessing without NLTK"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Split into words
    words = text.split()
    # Remove common English stop words
    stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                 'to', 'was', 'were', 'will', 'with'}
    return [word for word in words if word not in stop_words]

def generate_word_cloud(texts):
    """Generate word cloud visualization with basic preprocessing"""
    try:
        with st.spinner('Generating word cloud...'):
            # Preprocess texts
            processed_texts = []
            for text in texts:
                try:
                    tokens = preprocess_text(text)
                    processed_texts.append(' '.join(tokens))
                except Exception as e:
                    st.warning(f"Skipping text due to preprocessing error: {str(e)}")
            
            if not processed_texts:
                raise ValueError("No valid texts to process after preprocessing")
            
            combined_text = ' '.join(processed_texts)
            
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=200,
                contour_width=3,
                contour_color='steelblue'
            ).generate(combined_text)
            
            return wordcloud.to_array()
    except Exception as e:
        raise Exception(f"Error generating word cloud: {str(e)}")

def extract_key_phrases(texts, top_n=10):
    """Extract key phrases using TF-IDF"""
    try:
        with st.spinner('Extracting key phrases...'):
            tfidf = TfidfVectorizer(ngram_range=(2, 3), stop_words='english')
            tfidf_matrix = tfidf.fit_transform(texts)
            feature_names = tfidf.get_feature_names_out()
            
            # Get top phrases
            importance = np.array(tfidf_matrix.sum(axis=0)).ravel()
            top_indices = importance.argsort()[-top_n:][::-1]
            top_phrases = [(feature_names[i], float(importance[i])) for i in top_indices]
            return top_phrases
    except Exception as e:
        raise Exception(f"Error extracting key phrases: {str(e)}")

def analyze_text_length(texts):
    """Analyze text length distribution"""
    try:
        with st.spinner('Analyzing text length...'):
            lengths = [len(text.split()) for text in texts]
            return lengths
    except Exception as e:
        raise Exception(f"Error analyzing text length: {str(e)}")

def perform_topic_modeling(texts, n_topics=5):
    """Perform topic modeling using LDA"""
    try:
        with st.spinner('Performing topic modeling...'):
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(texts)
            
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda_output = lda.fit_transform(tfidf_matrix)
            
            feature_names = tfidf.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-10-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append((f"Topic {topic_idx + 1}", ', '.join(top_words)))
            
            return topics, lda_output
    except Exception as e:
        raise Exception(f"Error in topic modeling: {str(e)}")

def calculate_similarity_matrix(texts):
    """Calculate document similarity matrix"""
    try:
        with st.spinner('Calculating document similarity...'):
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
    except Exception as e:
        raise Exception(f"Error calculating similarity matrix: {str(e)}")

def advanced_search(texts, query, filters=None):
    """Perform advanced search with filtering capabilities"""
    try:
        with st.spinner('Performing advanced search...'):
            # Preprocess query
            query_tokens = set(preprocess_text(query))
            
            # Calculate TF-IDF for better matching
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(texts)
            
            # Calculate similarity scores
            query_vector = tfidf.transform([query])
            similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
            
            # Get matches with scores and indices
            matches = [(score, idx, text) for idx, (score, text) 
                      in enumerate(zip(similarity_scores, texts))]
            
            # Sort by similarity score
            matches.sort(reverse=True)
            
            # Apply filters if provided
            if filters:
                filtered_matches = []
                for score, idx, text in matches:
                    passes_filters = True
                    
                    for filter_type, filter_values in filters.items():
                        if filter_type == 'min_length' and filter_values:
                            if len(text.split()) < filter_values:
                                passes_filters = False
                                break
                    
                    if passes_filters:
                        filtered_matches.append((score, idx, text))
                matches = filtered_matches
            
            return matches
    except Exception as e:
        raise Exception(f"Error in advanced search: {str(e)}")
