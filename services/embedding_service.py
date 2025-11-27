"""
Embedding Service
Handles Gemini embedding model integration for semantic search
"""

import os
import asyncio
from typing import Optional, List, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Gemini embedding model
_embedding_model: Optional[Any] = None


def initialize_embedding_model() -> bool:
    """
    Initialize Gemini embedding model
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global _embedding_model
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print('❌ GEMINI_API_KEY not configured for embeddings')
        return False

    try:
        genai.configure(api_key=api_key)
        # Use text-embedding-004 model (768 dimensions)
        _embedding_model = genai.GenerativeModel('text-embedding-004')
        print('✅ Gemini embedding model initialized')
        return True
    except Exception as error:
        print(f'❌ Error initializing embedding model: {error}')
        return False


async def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate embedding vector for text using Gemini
    
    Args:
        text: Text to generate embedding for
    
    Returns:
        Optional[List[float]]: Embedding vector (768 dimensions) or None on error
    """
    if not text or not text.strip():
        return None

    # Re-initialize if model is null
    if not _embedding_model and not initialize_embedding_model():
        print('⚠️  Embedding model not configured')
        return None

    try:
        # Run synchronous embedding call in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model='models/text-embedding-004',
                content=text,
                task_type='RETRIEVAL_DOCUMENT'
            )
        )
        
        if result and 'embedding' in result:
            embedding = result['embedding']
            if embedding and len(embedding) > 0:
                return embedding
        
        print('⚠️  Empty embedding returned from Gemini')
        return None
    except Exception as error:
        print(f'❌ Error generating embedding: {error}')
        return None


async def generate_query_embedding(text: str) -> Optional[List[float]]:
    """
    Generate embedding for query text (optimized for retrieval)
    
    Args:
        text: Query text to generate embedding for
    
    Returns:
        Optional[List[float]]: Embedding vector (768 dimensions) or None on error
    """
    if not text or not text.strip():
        return None

    # Re-initialize if model is null
    if not _embedding_model and not initialize_embedding_model():
        print('⚠️  Embedding model not configured')
        return None

    try:
        # Run synchronous embedding call in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model='models/text-embedding-004',
                content=text,
                task_type='RETRIEVAL_QUERY'
            )
        )
        
        if result and 'embedding' in result:
            embedding = result['embedding']
            if embedding and len(embedding) > 0:
                return embedding
        
        print('⚠️  Empty embedding returned from Gemini')
        return None
    except Exception as error:
        print(f'❌ Error generating query embedding: {error}')
        return None


# Initialize on module load
if not initialize_embedding_model():
    print('⚠️  Embedding model not initialized. Memory retrieval will fail.')

