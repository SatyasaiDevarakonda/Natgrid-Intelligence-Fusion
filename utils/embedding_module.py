"""
NATGRID Semantic Search Module
Uses Sentence Transformers for embedding generation and similarity search
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """Semantic search engine for intelligence reports"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the semantic search engine
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.documents = None
        self.document_ids = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def index_documents(self, documents: List[str], document_ids: Optional[List[str]] = None):
        """
        Index documents by computing their embeddings
        
        Args:
            documents: List of document texts
            document_ids: Optional list of document identifiers
        """
        if not documents:
            logger.warning("No documents to index")
            return
        
        logger.info(f"Indexing {len(documents)} documents...")
        
        self.documents = documents
        self.document_ids = document_ids if document_ids else [str(i) for i in range(len(documents))]
        
        # Compute embeddings
        self.embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Indexed {len(documents)} documents with embedding shape: {self.embeddings.shape}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of dicts with document_id, text, and similarity score
        """
        if self.embeddings is None:
            logger.error("No documents indexed. Call index_documents first.")
            return []
        
        # Compute query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document_id': self.document_ids[idx],
                'text': self.documents[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def get_similar_documents(self, document_idx: int, top_k: int = 5) -> List[Dict]:
        """
        Find documents similar to a given document
        
        Args:
            document_idx: Index of the source document
            top_k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        if self.embeddings is None:
            logger.error("No documents indexed")
            return []
        
        if document_idx >= len(self.embeddings):
            logger.error(f"Invalid document index: {document_idx}")
            return []
        
        # Get source embedding
        source_embedding = self.embeddings[document_idx].reshape(1, -1)
        
        # Compute similarities
        similarities = cosine_similarity(source_embedding, self.embeddings)[0]
        
        # Get top-k indices (excluding the source document)
        top_indices = np.argsort(similarities)[-(top_k + 1):][::-1]
        top_indices = [idx for idx in top_indices if idx != document_idx][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document_id': self.document_ids[idx],
                'text': self.documents[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def cluster_documents(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Cluster documents using K-means
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Dict mapping cluster id to list of document ids
        """
        from sklearn.cluster import KMeans
        
        if self.embeddings is None:
            logger.error("No documents indexed")
            return {}
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.document_ids[idx])
        
        return clusters
    
    def save_index(self, filepath: str):
        """Save the index to disk"""
        data = {
            'embeddings': self.embeddings,
            'documents': self.documents,
            'document_ids': self.document_ids
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load the index from disk"""
        if not os.path.exists(filepath):
            logger.error(f"Index file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.documents = data['documents']
        self.document_ids = data['document_ids']
        logger.info(f"Index loaded from {filepath}")
        return True


def get_search_engine() -> SemanticSearchEngine:
    """Factory function to get search engine instance"""
    return SemanticSearchEngine()


if __name__ == "__main__":
    # Test the semantic search engine
    engine = SemanticSearchEngine()
    
    test_documents = [
        "Intelligence indicates Abdul Karim planning coordinated attack in Mumbai.",
        "Customs officials seized contraband at Mundra Port.",
        "Cricket match between India and Australia scheduled at Delhi.",
        "Tech company announced expansion plans in Bangalore.",
        "Weapons smuggling detected at border checkpoint.",
        "Cyber attack detected targeting government systems."
    ]
    
    test_ids = [f"DOC{i:03d}" for i in range(len(test_documents))]
    
    # Index documents
    engine.index_documents(test_documents, test_ids)
    
    # Test search
    print("=" * 50)
    print("SEMANTIC SEARCH TEST")
    print("=" * 50)
    
    query = "weapons smuggling border"
    print(f"\nQuery: {query}")
    results = engine.search(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['document_id']} (Score: {result['similarity']:.3f})")
        print(f"   {result['text'][:100]}...")
