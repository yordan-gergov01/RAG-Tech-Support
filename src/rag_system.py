"""RAG System Class with FAISS"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama
from typing import Dict, List

class RAGSystem:
    def __init__(self, 
                 vector_db_path: str = "vector_db",
                 model_name: str = "llama3.2:3b"):
        """
        Initialize RAG System with FAISS

        Args:
            vector_db_path: Path to FAISS index and metadata
            model_name: Ollama model name
        """
        self.model_name = model_name
        self.vector_db_path = vector_db_path

        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load FAISS index
        index_path = os.path.join(vector_db_path, 'faiss_index.bin')
        self.index = faiss.read_index(index_path)

        # Load metadata
        metadata_path = os.path.join(vector_db_path, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadatas = pickle.load(f)

        print(f"RAG System initialized")
        print(f"  Vector DB: {vector_db_path}")
        print(f"  Documents: {self.index.ntotal}")
        print(f"  Model: {model_name}")

    def search(self, query: str, n_results: int = 3) -> Dict:
        """Search knowledge base"""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        # Search in FAISS
        distances, indices = self.index.search(query_embedding, n_results)

        # Get metadata
        results = {
            'distances': distances[0].tolist(),
            'indices': indices[0].tolist(),
            'documents': [],
            'metadatas': []
        }

        for idx in indices[0]:
            metadata = self.metadatas[idx]
            results['documents'].append(metadata['text'])
            results['metadatas'].append(metadata)

        return results

    def query(self, question: str, n_results: int = 3, verbose: bool = False) -> Dict:
        """Full RAG pipeline"""
        # Retrieve
        search_results = self.search(question, n_results=n_results)

        retrieved_docs = search_results['documents']
        retrieved_metadata = search_results['metadatas']
        distances = search_results['distances']

        # Prepare context
        context = "\n\n---\n\n".join([
            f"Document {i+1} (from {meta['product']} - {meta['category']}):\n{doc}"
            for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadata))
        ])

        # Generate
        prompt = f"""You are a helpful technical support assistant.

Use the following documentation to answer the user's question. 
If the answer is not in the documentation, say so.
Be concise and helpful.

Documentation:
{context}

User Question: {question}

Answer:"""

        response = ollama.generate(
            model=self.model_name,
            prompt=prompt
        )

        answer = response['response']

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'retrieved_metadata': retrieved_metadata,
            'distances': distances,
            'context': context
        }
