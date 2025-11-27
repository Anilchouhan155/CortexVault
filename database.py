"""
Database Module
Handles Supabase database operations with JSON file fallback
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize Supabase client (if credentials are available)
supabase: Optional[Client] = None
use_supabase = False

if os.getenv('SUPABASE_URL') and os.getenv('SUPABASE_SERVICE_ROLE_KEY'):
    try:
        supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        )
        use_supabase = True
        print('✅ Supabase database initialized')
    except Exception as error:
        print(f'⚠️  Failed to initialize Supabase, falling back to JSON storage: {error}')
        use_supabase = False
else:
    print('⚠️  Supabase credentials not found, using JSON file storage')
    print('   Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to use Supabase')

# Fallback JSON storage
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / 'conversations.json'
BACKUP_FILE = BASE_DIR / 'conversations.backup.json'


# Pydantic models for validation
class ConversationData(BaseModel):
    """Pydantic model for conversation data validation"""
    phone: str = Field(..., min_length=1, description="Phone number")
    userName: str = Field(default='User', description="User name")
    userMessage: str = Field(..., min_length=1, description="User message")
    aiResponse: str = Field(..., min_length=1, description="AI response")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp")
    messageType: str = Field(default='text', description="Message type")
    voiceMetadata: Optional[Dict[str, Any]] = Field(default=None, description="Voice metadata")

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v: str) -> str:
        """Validate phone number"""
        if not v or not v.strip():
            raise ValueError('Phone number is required')
        return v.strip()

    @field_validator('userMessage', 'aiResponse')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message content"""
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()


def initialize_data_file() -> None:
    """Initialize data file if it doesn't exist (fallback only)"""
    if not DATA_FILE.exists():
        try:
            initial_data = {'conversations': []}
            DATA_FILE.write_text(json.dumps(initial_data, indent=2))
            print('📁 Conversations database initialized (JSON fallback)')
        except Exception as error:
            print(f'❌ Error initializing data file: {error}')
            raise error


def read_data() -> Dict[str, Any]:
    """Read data from file (fallback only)"""
    try:
        initialize_data_file()
        
        if not DATA_FILE.exists():
            return {'conversations': []}

        file_content = DATA_FILE.read_text(encoding='utf-8')
        
        if not file_content or not file_content.strip():
            return {'conversations': []}

        data = json.loads(file_content)
        
        if not data or not isinstance(data, dict) or not isinstance(data.get('conversations'), list):
            return {'conversations': []}

        return data
    except Exception as error:
        print(f'❌ Error reading data file: {error}')
        return {'conversations': []}


def write_data(data: Dict[str, Any]) -> bool:
    """Write data to file (fallback only)"""
    if not data or not isinstance(data, dict) or not isinstance(data.get('conversations'), list):
        return False

    try:
        temp_file = DATA_FILE.with_suffix('.tmp')
        temp_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
        temp_file.replace(DATA_FILE)
        return True
    except Exception as error:
        print(f'❌ Error writing data file: {error}')
        return False


async def get_or_create_user(phone: str, user_name: str) -> Optional[str]:
    """
    Get or create user in Supabase
    
    Args:
        phone: Phone number
        user_name: User name
    
    Returns:
        Optional[str]: User ID or None on error
    """
    if not use_supabase:
        return None

    if not phone or not phone.strip():
        return None

    try:
        # Try to find existing user
        response = supabase.table('users').select('id').eq('phone_number', phone).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            existing_user = response.data[0]
            # Update last_active
            supabase.table('users').update({
                'last_active': datetime.now().isoformat()
            }).eq('id', existing_user['id']).execute()
            
            return existing_user['id']

        # Create new user
        new_user_response = supabase.table('users').insert({
            'phone_number': phone,
            'whatsapp_verified': True,
            'last_active': datetime.now().isoformat()
        }).execute()

        if not new_user_response.data or len(new_user_response.data) == 0:
            return None

        new_user = new_user_response.data[0]

        # Create user profile
        if new_user and user_name:
            supabase.table('user_profiles').insert({
                'user_id': new_user['id'],
                'first_name': user_name
            }).execute()

        return new_user['id']
    except Exception as error:
        print(f'❌ Error in get_or_create_user: {error}')
        return None


async def save_conversation(conversation_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Save a new conversation
    
    Args:
        conversation_data: Dict with phone, userName, userMessage, aiResponse, timestamp, etc.
    
    Returns:
        Optional[Dict[str, Any]]: Saved conversation data or None on error
    """
    # Validate input data using Pydantic
    if not conversation_data:
        print('❌ Conversation data is required')
        return None

    try:
        # Validate with Pydantic model
        validated_data = ConversationData(**conversation_data)
    except Exception as validation_error:
        print(f'❌ Invalid conversation data: {validation_error}')
        return None

    phone = validated_data.phone
    user_name = validated_data.userName
    user_message = validated_data.userMessage
    ai_response = validated_data.aiResponse
    timestamp = validated_data.timestamp
    message_type = validated_data.messageType
    voice_metadata = validated_data.voiceMetadata

    # Use Supabase if available
    if use_supabase:
        try:
            # Get or create user
            user_id = await get_or_create_user(phone, user_name)
            
            if not user_id:
                print('❌ Failed to get or create user')
                # Fall through to JSON fallback
            else:
                # Save conversation to Supabase
                conversation_to_save = {
                    'user_id': user_id,
                    'user_message': user_message,
                    'ai_response': ai_response,
                    'message_type': message_type,
                    'created_at': timestamp
                }

                # Add voice metadata if available (store as JSONB in conversation_context)
                if voice_metadata:
                    conversation_to_save['conversation_context'] = {'voice': voice_metadata}

                response = supabase.table('conversations').insert(conversation_to_save).execute()

                if not response.data or len(response.data) == 0:
                    print('❌ Failed to save conversation to Supabase')
                    # Fall through to JSON fallback
                else:
                    conversation = response.data[0]
                    print(f'💾 Conversation saved to Supabase: {conversation["id"]}')
                    
                    # Return in same format as JSON version for compatibility
                    return {
                        'id': str(conversation['id']),
                        'phone': phone,
                        'userName': user_name,
                        'userMessage': user_message,
                        'aiResponse': ai_response,
                        'timestamp': timestamp
                    }
        except Exception as error:
            print(f'❌ Error saving to Supabase, falling back to JSON: {error}')
            # Fall through to JSON fallback

    # Fallback to JSON storage
    try:
        data = read_data()
        
        import random
        import string
        conversation = {
            'id': f"{int(datetime.now().timestamp() * 1000)}{''.join(random.choices(string.ascii_lowercase + string.digits, k=7))}",
            'phone': phone,
            'userName': user_name,
            'userMessage': user_message,
            'aiResponse': ai_response,
            'timestamp': timestamp
        }

        # Limit conversation history (keep last 10000)
        if len(data['conversations']) >= 10000:
            print('⚠️  Conversation history limit reached, removing oldest entries')
            data['conversations'] = data['conversations'][-9000:]
        
        data['conversations'].append(conversation)
        
        write_success = write_data(data)
        
        if not write_success:
            print('❌ Failed to write conversation to file')
            return None
        
        print(f'💾 Conversation saved to JSON: {conversation["id"]}')
        return conversation
    except Exception as error:
        print(f'❌ Error saving conversation: {error}')
        return None


async def get_conversations() -> List[Dict[str, Any]]:
    """
    Get all conversations
    
    Returns:
        List[Dict[str, Any]]: List of conversation dictionaries
    """
    try:
        if use_supabase:
            try:
                # First get conversations with user_id
                response = supabase.table('conversations').select(
                    'id, user_id, user_message, ai_response, created_at'
                ).order('created_at', desc=True).limit(1000).execute()

                if not response.data:
                    return []

                conversations = response.data

                # Get unique user IDs
                user_ids = list(set([conv['user_id'] for conv in conversations]))
                
                # Get users and their phone numbers
                users_response = supabase.table('users').select('id, phone_number').in_('id', user_ids).execute()
                users = users_response.data if users_response.data else []

                # Get user profiles for names
                profiles_response = supabase.table('user_profiles').select('user_id, first_name').in_('user_id', user_ids).execute()
                profiles = profiles_response.data if profiles_response.data else []

                # Create lookup maps
                user_map = {u['id']: u['phone_number'] for u in users}
                profile_map = {p['user_id']: p['first_name'] for p in profiles}

                # Transform to match JSON format
                return [{
                    'id': str(conv['id']),
                    'phone': user_map.get(conv['user_id'], 'unknown'),
                    'userName': profile_map.get(conv['user_id'], 'User'),
                    'userMessage': conv['user_message'],
                    'aiResponse': conv['ai_response'],
                    'timestamp': conv['created_at']
                } for conv in conversations]
            except Exception as error:
                print(f'❌ Error fetching from Supabase, falling back to JSON: {error}')
                # Fall through to JSON fallback

        # Fallback to JSON
        data = read_data()
        
        if not data.get('conversations') or not isinstance(data['conversations'], list):
            return []

        conversations = data['conversations']
        
        # Filter and sort
        valid_conversations = []
        for conv in conversations:
            if not (conv and conv.get('id') and conv.get('phone') and conv.get('timestamp')):
                continue
            try:
                # Try to parse timestamp to validate
                timestamp_str = conv['timestamp'].replace('Z', '+00:00')
                datetime.fromisoformat(timestamp_str)
                valid_conversations.append(conv)
            except (ValueError, AttributeError):
                continue
        
        # Sort by timestamp descending
        def get_timestamp(conv):
            try:
                return datetime.fromisoformat(conv['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return datetime.min
        
        valid_conversations.sort(key=get_timestamp, reverse=True)
        
        return valid_conversations
    except Exception as error:
        print(f'❌ Error getting conversations: {error}')
        return []


async def get_recent_conversations(phone: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get recent conversations for a specific user
    
    Args:
        phone: Phone number
        limit: Maximum number of conversations to return
    
    Returns:
        List[Dict[str, Any]]: List of conversation dictionaries
    """
    if not phone or not isinstance(phone, str):
        print('⚠️  Invalid phone number provided to get_recent_conversations')
        return []

    try:

        limit_num = min(int(limit) if limit else 5, 50)

        if use_supabase:
            try:
                # First get user ID
                user_response = supabase.table('users').select('id').eq('phone_number', phone).limit(1).execute()
                
                if not user_response.data or len(user_response.data) == 0:
                    return []

                user = user_response.data[0]

                # Get conversations for this user
                conversations_response = supabase.table('conversations').select(
                    'id, user_message, ai_response, created_at'
                ).eq('user_id', user['id']).order('created_at', desc=True).limit(limit_num).execute()

                if not conversations_response.data:
                    return []

                # Transform to match JSON format
                return [{
                    'id': str(conv['id']),
                    'phone': phone,
                    'userName': 'User',  # Will be populated from user_profiles if needed
                    'userMessage': conv['user_message'],
                    'aiResponse': conv['ai_response'],
                    'timestamp': conv['created_at']
                } for conv in conversations_response.data]
            except Exception as error:
                print(f'❌ Error fetching from Supabase, falling back to JSON: {error}')
                # Fall through to JSON fallback

        # Fallback to JSON
        data = read_data()
        
        if not data.get('conversations') or not isinstance(data['conversations'], list):
            return []

        user_conversations = [
            conv for conv in data['conversations']
            if conv and conv.get('phone') and conv['phone'].strip() == phone.strip()
            and conv.get('timestamp')
        ]
        
        # Sort by timestamp descending
        def get_timestamp(conv):
            try:
                return datetime.fromisoformat(conv['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return datetime.min
        
        user_conversations.sort(key=get_timestamp, reverse=True)
        
        return user_conversations[:limit_num]
    except Exception as error:
        print(f'❌ Error getting recent conversations: {error}')
        return []


# ============================================
# Memory System Functions
# ============================================

async def save_memory(
    user_id: str,
    content: str,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Save a memory with embedding to Supabase
    
    Args:
        user_id: User ID (phone number or UUID)
        content: Memory content text
        embedding: Embedding vector (768 dimensions)
        metadata: Optional metadata dict (type, topic, tags, etc.)
    
    Returns:
        Optional[str]: Memory ID or None on error
    """
    if not use_supabase:
        return None

    if not user_id or not content or not content.strip():
        print('⚠️  Invalid memory data: user_id and content required')
        return None

    try:
        memory_data = {
            'user_id': user_id,
            'content': content.strip(),
            'metadata': metadata or {}
        }

        # Add embedding if provided
        if embedding:
            memory_data['embedding'] = embedding

        response = supabase.table('memories').insert(memory_data).execute()

        if not response.data or len(response.data) == 0:
            print('❌ Failed to save memory')
            return None

        memory_id = response.data[0]['id']
        print(f'💾 Memory saved: {memory_id}')
        return str(memory_id)
    except Exception as error:
        print(f'❌ Error saving memory: {error}')
        return None


async def retrieve_memories(
    user_id: str,
    query_embedding: List[float],
    limit: int = 5,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant memories using vector similarity search
    
    Args:
        user_id: User ID to filter memories
        query_embedding: Query embedding vector (768 dimensions)
        limit: Maximum number of memories to return
        similarity_threshold: Minimum cosine similarity (0-1)
    
    Returns:
        List[Dict[str, Any]]: List of relevant memories
    """
    if not use_supabase:
        return []

    if not user_id or not query_embedding:
        return []

    if len(query_embedding) != 768:
        print('⚠️  Invalid embedding dimension (expected 768)')
        return []

    try:
        # Use Supabase RPC function for efficient vector similarity search
        # This leverages the HNSW index for fast retrieval
        try:
            # Call the search_memories RPC function
            # Note: threshold of 0.0 means no threshold filtering (return top N)
            actual_threshold = similarity_threshold if similarity_threshold > 0.0 else 0.0
            
            response = supabase.rpc(
                'search_memories',
                {
                    'p_user_id': user_id,
                    'p_query_embedding': query_embedding,
                    'p_limit': limit,
                    'p_similarity_threshold': actual_threshold
                }
            ).execute()

            if not response.data:
                return []

            # Format results
            memories = []
            for memory in response.data:
                memories.append({
                    'id': str(memory['id']),
                    'content': memory['content'],
                    'metadata': memory.get('metadata', {}),
                    'similarity': float(memory.get('similarity', 0.0)),
                    'created_at': memory.get('created_at')
                })

            return memories
        except Exception as rpc_error:
            # Fallback to Python-based calculation if RPC fails
            print(f'⚠️  RPC function failed, falling back to Python calculation: {rpc_error}')
            
            # Get all memories for user and calculate similarity in Python
            response = supabase.table('memories').select(
                'id, content, embedding, metadata, created_at'
            ).eq('user_id', user_id).execute()

            if not response.data:
                return []

            # Calculate cosine similarity for each memory
            import math

            def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
                """Calculate cosine similarity between two vectors"""
                if len(vec1) != len(vec2):
                    return 0.0
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                magnitude1 = math.sqrt(sum(a * a for a in vec1))
                magnitude2 = math.sqrt(sum(a * a for a in vec2))
                if magnitude1 == 0 or magnitude2 == 0:
                    return 0.0
                return dot_product / (magnitude1 * magnitude2)

            memories_with_similarity = []
            for memory in response.data:
                if not memory.get('embedding'):
                    continue
                
                embedding = memory['embedding']
                similarity = cosine_similarity(query_embedding, embedding)
                
                # If threshold is 0.0, include all memories (no filtering)
                # Otherwise, filter by threshold
                if similarity_threshold == 0.0 or similarity >= similarity_threshold:
                    memories_with_similarity.append({
                        'id': str(memory['id']),
                        'content': memory['content'],
                        'metadata': memory.get('metadata', {}),
                        'similarity': similarity,
                        'created_at': memory.get('created_at')
                    })

            # Sort by similarity descending
            memories_with_similarity.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top N memories
            return memories_with_similarity[:limit]
    except Exception as error:
        print(f'❌ Error retrieving memories: {error}')
        return []


async def retrieve_memories_hybrid(
    user_id: str,
    user_message: str,
    query_embedding: Optional[List[float]] = None,
    limit: int = 5,
    similarity_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Hybrid search: combines semantic (vector) and keyword search
    
    Args:
        user_id: User ID to filter memories
        user_message: Original user message for keyword extraction
        query_embedding: Query embedding vector (768 dimensions) - generated if not provided
        limit: Maximum number of memories to return
        similarity_threshold: Minimum cosine similarity for vector search
    
    Returns:
        List[Dict[str, Any]]: List of relevant memories with boosted exact matches
    """
    if not use_supabase:
        return []
    
    if not user_id or not user_message:
        return []
    
    import re
    from services.embedding_service import generate_query_embedding
    
    # Extract entities from query (names, phone numbers, etc.)
    # Phone numbers: digits with 7+ digits
    phone_pattern = r'\b\d{7,}\b'
    phone_numbers = re.findall(phone_pattern, user_message)
    
    # Names: capitalized words (simple heuristic)
    # Look for patterns like "riya", "Riya", "riya's", etc.
    name_pattern = r'\b([A-Z][a-z]+)\b|\b([a-z]{3,})\b'
    potential_names = []
    for match in re.finditer(name_pattern, user_message.lower()):
        word = match.group(0).lower()
        # Filter out common words
        if word not in ['the', 'what', 'where', 'when', 'who', 'how', 'can', 'you', 'tell', 'me', 'is', 'are', 'was', 'were']:
            potential_names.append(word)
    
    # Combine semantic and keyword results
    all_memories = {}
    
    # 1. Semantic search (vector)
    semantic_memories = []
    if query_embedding or True:  # Always try semantic search
        try:
            if not query_embedding:
                query_embedding = await generate_query_embedding(user_message)
            
            if query_embedding:
                semantic_memories = await retrieve_memories(
                    user_id=user_id,
                    query_embedding=query_embedding,
                    limit=limit * 2,  # Get more for merging
                    similarity_threshold=similarity_threshold
                )
        except Exception as error:
            print(f'⚠️  Semantic search error: {error}')
    
    # Add semantic results with base score
    for mem in semantic_memories:
        mem_id = mem.get('id')
        if mem_id:
            # Store with similarity as base score
            all_memories[mem_id] = {
                **mem,
                'search_score': mem.get('similarity', 0.0),
                'match_type': 'semantic'
            }
    
    # 2. Keyword search for exact/partial matches
    keyword_memories = []
    try:
        # Search for phone numbers
        if phone_numbers:
            for phone in phone_numbers:
                response = supabase.table('memories').select(
                    'id, content, metadata, created_at'
                ).eq('user_id', user_id).ilike('content', f'%{phone}%').limit(10).execute()
                
                if response.data:
                    for mem in response.data:
                        keyword_memories.append({
                            'id': str(mem['id']),
                            'content': mem['content'],
                            'metadata': mem.get('metadata', {}),
                            'similarity': 0.0,  # Will be boosted
                            'created_at': mem.get('created_at')
                        })
        
        # Search for names
        if potential_names:
            for name in potential_names:
                # Search for name in content (case-insensitive)
                response = supabase.table('memories').select(
                    'id, content, metadata, created_at'
                ).eq('user_id', user_id).ilike('content', f'%{name}%').limit(10).execute()
                
                if response.data:
                    for mem in response.data:
                        keyword_memories.append({
                            'id': str(mem['id']),
                            'content': mem['content'],
                            'metadata': mem.get('metadata', {}),
                            'similarity': 0.0,  # Will be boosted
                            'created_at': mem.get('created_at')
                        })
    except Exception as error:
        print(f'⚠️  Keyword search error: {error}')
    
    # Add keyword results with boosted scores
    for mem in keyword_memories:
        mem_id = mem.get('id')
        if mem_id:
            # Boost exact matches significantly
            content_lower = mem.get('content', '').lower()
            message_lower = user_message.lower()
            
            # Check for exact phrase match
            if any(name in content_lower for name in potential_names if name):
                boost = 0.3  # Significant boost for name matches
            elif any(phone in content_lower for phone in phone_numbers):
                boost = 0.4  # Even higher boost for phone number matches
            else:
                boost = 0.2  # Moderate boost for partial matches
            
            if mem_id in all_memories:
                # Already exists from semantic search - boost it
                all_memories[mem_id]['search_score'] = min(1.0, all_memories[mem_id]['search_score'] + boost)
                all_memories[mem_id]['match_type'] = 'hybrid'
            else:
                # New from keyword search
                all_memories[mem_id] = {
                    **mem,
                    'search_score': 0.5 + boost,  # Base score + boost
                    'match_type': 'keyword'
                }
    
    # Convert to list and sort by search score
    result_memories = list(all_memories.values())
    result_memories.sort(key=lambda x: x.get('search_score', 0.0), reverse=True)
    
    # Return top N, ensuring we prioritize hybrid matches
    # First, get hybrid matches, then semantic-only
    hybrid_matches = [m for m in result_memories if m.get('match_type') == 'hybrid']
    semantic_only = [m for m in result_memories if m.get('match_type') == 'semantic']
    keyword_only = [m for m in result_memories if m.get('match_type') == 'keyword']
    
    # Combine: hybrid first, then keyword, then semantic
    final_results = (hybrid_matches + keyword_only + semantic_only)[:limit]
    
    # Clean up the search_score field (keep similarity for compatibility)
    for mem in final_results:
        if 'search_score' in mem:
            mem['similarity'] = mem.pop('search_score')
        if 'match_type' in mem:
            mem.pop('match_type')
    
    return final_results


async def get_user_session(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get or create user session
    
    Args:
        user_id: User ID (phone number)
    
    Returns:
        Optional[Dict[str, Any]]: Session data or None on error
    """
    if not use_supabase:
        return None

    if not user_id:
        return None

    try:
        # Try to get existing session
        response = supabase.table('user_sessions').select('*').eq('user_id', user_id).limit(1).execute()
        
        if response.data and len(response.data) > 0:
            session = response.data[0]
            return {
                'id': str(session['id']),
                'user_id': session['user_id'],
                'session_buffer': session.get('session_buffer', []),
                'last_message_time': session.get('last_message_time'),
                'created_at': session.get('created_at'),
                'updated_at': session.get('updated_at')
            }

        # Create new session
        new_session = {
            'user_id': user_id,
            'session_buffer': [],
            'last_message_time': datetime.now().isoformat()
        }
        
        response = supabase.table('user_sessions').insert(new_session).execute()
        
        if not response.data or len(response.data) == 0:
            return None

        session = response.data[0]
        return {
            'id': str(session['id']),
            'user_id': session['user_id'],
            'session_buffer': session.get('session_buffer', []),
            'last_message_time': session.get('last_message_time'),
            'created_at': session.get('created_at'),
            'updated_at': session.get('updated_at')
        }
    except Exception as error:
        print(f'❌ Error getting user session: {error}')
        return None


async def update_session(
    user_id: str,
    message: Dict[str, Any],
    max_buffer_size: int = 6
) -> bool:
    """
    Update session buffer with new message
    
    Args:
        user_id: User ID (phone number)
        message: Message dict with userMessage and aiResponse
        max_buffer_size: Maximum number of messages in buffer
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not use_supabase:
        return False

    if not user_id or not message:
        return False

    try:
        # Get current session
        session = await get_user_session(user_id)
        if not session:
            return False

        # Get current buffer
        buffer = session.get('session_buffer', [])
        if not isinstance(buffer, list):
            buffer = []

        # Add new message to buffer
        buffer.append({
            'userMessage': message.get('userMessage', ''),
            'aiResponse': message.get('aiResponse', ''),
            'timestamp': message.get('timestamp', datetime.now().isoformat())
        })

        # Keep only last N messages
        if len(buffer) > max_buffer_size:
            buffer = buffer[-max_buffer_size:]

        # Update session
        update_data = {
            'session_buffer': buffer,
            'last_message_time': datetime.now().isoformat()
        }

        response = supabase.table('user_sessions').update(update_data).eq('user_id', user_id).execute()
        
        return response.data is not None
    except Exception as error:
        print(f'❌ Error updating session: {error}')
        return False


async def finalize_session(user_id: str) -> bool:
    """
    Finalize session and clear buffer (called when session ends)
    
    Args:
        user_id: User ID (phone number)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not use_supabase:
        return False

    if not user_id:
        return False

    try:
        # Clear session buffer
        update_data = {
            'session_buffer': [],
            'last_message_time': datetime.now().isoformat()
        }

        response = supabase.table('user_sessions').update(update_data).eq('user_id', user_id).execute()
        
        return response.data is not None
    except Exception as error:
        print(f'❌ Error finalizing session: {error}')
        return False

