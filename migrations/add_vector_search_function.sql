-- Vector Search Function for Memory Retrieval
-- Uses pgvector for efficient similarity search with HNSW index

-- Create function for vector similarity search
CREATE OR REPLACE FUNCTION search_memories(
    p_user_id TEXT,
    p_query_embedding vector(768),
    p_limit INTEGER DEFAULT 5,
    p_similarity_threshold FLOAT DEFAULT 0.5
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    similarity FLOAT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id,
        m.content,
        m.metadata,
        -- Calculate cosine similarity: 1 - cosine_distance
        -- cosine_distance uses <=> operator
        1 - (m.embedding <=> p_query_embedding) AS similarity,
        m.created_at
    FROM memories m
    WHERE 
        m.user_id = p_user_id
        AND m.embedding IS NOT NULL
        AND (1 - (m.embedding <=> p_query_embedding)) >= p_similarity_threshold
    ORDER BY m.embedding <=> p_query_embedding  -- Order by distance (ascending = most similar first)
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Add comment for documentation
COMMENT ON FUNCTION search_memories IS 'Efficient vector similarity search using pgvector HNSW index. Returns memories sorted by cosine similarity.';

