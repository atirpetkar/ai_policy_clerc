import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import io
import re
from utils.vector_store import VectorStore
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_basic_statistics(df: pd.DataFrame) -> Dict:
    """Generate basic statistics for the DataFrame."""
    try:
        logger.info("Generating basic statistics for DataFrame")
        
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': {col: int(df[col].isna().sum()) for col in df.columns}
        }
        
        logger.info(f"Successfully generated statistics for DataFrame with {stats['total_rows']} rows and {stats['total_columns']} columns")
        return stats
        
    except Exception as e:
        logger.error(f"Error generating basic statistics: {str(e)}", exc_info=True)
        raise Exception(f"Failed to generate basic statistics: {str(e)}")

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string."""
    words = re.findall(r'\w+|[^\w\s]', text)
    return int(len(words) * 1.2)

def chunk_text(text: str, max_tokens: int = 3000, overlap: int = 50) -> List[str]:
    """Split text into chunks while preserving context and staying within token limits."""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        if sentence_tokens > max_tokens:
            words = sentence.split()
            temp_chunk = []
            temp_tokens = 0
            
            for word in words:
                word_tokens = estimate_tokens(word)
                if temp_tokens + word_tokens > max_tokens:
                    chunks.append(' '.join(temp_chunk))
                    temp_chunk = []
                    temp_tokens = 0
                temp_chunk.append(word)
                temp_tokens += word_tokens
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
            continue
        
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(' '.join(current_chunk))
            overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) > 2 else ''
            current_chunk = []
            if overlap_text:
                current_chunk.append(overlap_text)
            current_tokens = estimate_tokens(' '.join(current_chunk))
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_csv_data(file: io.BytesIO, content_columns: List[str], metadata_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Process CSV data with flexible column mapping and token limit handling."""
    try:
        # Reset file pointer and read CSV
        file.seek(0)
        df = pd.read_csv(file)
        
        # Validate content columns
        if not content_columns:
            raise ValueError("No content columns specified")
        
        # Check for missing columns
        missing_columns = [col for col in content_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing content columns: {', '.join(missing_columns)}")
        
        # Initialize text processing
        processed_texts = []
        logger.info(f"Processing {len(df)} documents")
        
        # Process text content
        for idx, row in df.iterrows():
            combined_text = " ".join(str(row[col]) for col in content_columns 
                                   if not pd.isna(row[col]))
            
            if combined_text.strip():
                chunks = chunk_text(combined_text)
                processed_texts.extend(chunks)
            else:
                logger.warning(f"Empty text content in row {idx}")
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame()
        if metadata_columns:
            metadata_df = df[metadata_columns].copy()
            if len(processed_texts) > len(df):
                repeat_count = len(processed_texts) // len(df) + 1
                expanded_values = np.tile(metadata_df.values, (repeat_count, 1))
                metadata_df = pd.DataFrame(
                    expanded_values[:len(processed_texts)],
                    columns=metadata_df.columns
                )
        
        logger.info(f"Processed {len(processed_texts)} documents")
        return df, processed_texts, metadata_df
        
    except Exception as e:
        logger.error(f"Error processing CSV data: {str(e)}", exc_info=True)
        raise Exception(f"Failed to process CSV data: {str(e)}")
