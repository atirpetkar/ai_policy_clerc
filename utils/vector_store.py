import faiss
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []
        self.metadata = pd.DataFrame()
        self.dimension = 1536  # OpenAI ada-002 embedding dimension

    def create_index(self, embeddings: List[List[float]], texts: List[str], metadata_df: Optional[pd.DataFrame] = None):
        """Create FAISS index from embeddings with optional metadata."""
        self.texts = texts
        embeddings_np = np.array(embeddings).astype('float32')
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)
        
        if metadata_df is not None:
            self.metadata = metadata_df

    def search(self, query_embedding: List[float], k: int = 3, 
              filters: Optional[Dict] = None) -> Tuple[List[str], List[float], pd.DataFrame]:
        """Search for similar texts using query embedding with optional metadata filtering."""
        if self.index is None:
            raise ValueError("Index not initialized")

        query_np = np.array([query_embedding]).astype('float32')
        
        # Get more results initially if we have filters
        search_k = k * 3 if filters else k
        distances, indices = self.index.search(query_np, search_k)
        
        results = []
        scores = []
        filtered_metadata = []
        
        # Apply metadata filters if provided
        for idx, (distance, doc_idx) in enumerate(zip(distances[0], indices[0])):
            if len(results) >= k:
                break
                
            doc_metadata = self.metadata.iloc[doc_idx] if not self.metadata.empty else None
            
            # Check if document passes all filters
            passes_filters = True
            if filters and doc_metadata is not None:
                for key, value in filters.items():
                    if key in doc_metadata:
                        if isinstance(value, (list, tuple)):
                            if doc_metadata[key] not in value:
                                passes_filters = False
                                break
                        elif doc_metadata[key] != value:
                            passes_filters = False
                            break
            
            if passes_filters:
                results.append(self.texts[doc_idx])
                scores.append(distance)
                if doc_metadata is not None:
                    filtered_metadata.append(doc_metadata)
        
        metadata_df = pd.DataFrame(filtered_metadata) if filtered_metadata else pd.DataFrame()
        return results, scores, metadata_df

    def get_metadata_fields(self) -> List[str]:
        """Get available metadata fields."""
        return list(self.metadata.columns) if not self.metadata.empty else []

    def get_unique_values(self, field: str) -> List:
        """Get unique values for a metadata field."""
        if field in self.metadata.columns:
            return sorted(self.metadata[field].unique().tolist())
        return []
