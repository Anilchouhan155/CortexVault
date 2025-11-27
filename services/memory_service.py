"""
Memory Service
Handles session management, memory extraction, and retrieval
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from database import (
    get_user_session,
    update_session,
    finalize_session,
    save_memory,
    retrieve_memories,
    retrieve_memories_hybrid
)
from services.embedding_service import generate_embedding, generate_query_embedding

# Load environment variables
load_dotenv()

# Session timeout (1 hour)
SESSION_TIMEOUT_HOURS = 1


async def check_session_status(user_id: str) -> Dict[str, Any]:
    """
    Check if session is active or needs to be finalized
    
    Args:
        user_id: User ID (phone number)
    
    Returns:
        Dict with session_status ('active' or 'expired'), session data, and needs_finalization flag
    """
    if not user_id:
        return {
            'session_status': 'expired',
            'session': None,
            'needs_finalization': False
        }

    try:
        session = await get_user_session(user_id)
    except Exception as error:
        print(f'⚠️  Error getting session (corrupted?): {error}')
        # Try to recover by creating new session
        try:
            from database import finalize_session
            await finalize_session(user_id)
            session = await get_user_session(user_id)
        except Exception:
            return {
                'session_status': 'active',
                'session': None,
                'needs_finalization': False
            }
    
    if not session:
        return {
            'session_status': 'active',
            'session': None,
            'needs_finalization': False
        }

    last_message_time = session.get('last_message_time')
    if not last_message_time:
        return {
            'session_status': 'active',
            'session': session,
            'needs_finalization': False
        }

    # Parse timestamp
    try:
        if isinstance(last_message_time, str):
            last_time = datetime.fromisoformat(last_message_time.replace('Z', '+00:00'))
        else:
            last_time = last_message_time
        
        # Check if session expired (more than 1 hour)
        time_diff = datetime.now(last_time.tzinfo) - last_time
        is_expired = time_diff > timedelta(hours=SESSION_TIMEOUT_HOURS)
        
        return {
            'session_status': 'expired' if is_expired else 'active',
            'session': session,
            'needs_finalization': is_expired
        }
    except Exception as error:
        print(f'⚠️  Error checking session status: {error}')
        # Session data might be corrupted, reset it
        try:
            from database import finalize_session
            await finalize_session(user_id)
        except Exception:
            pass
        return {
            'session_status': 'active',
            'session': None,
            'needs_finalization': False
        }


async def get_short_term_memory(user_id: str) -> List[Dict[str, Any]]:
    """
    Get short-term memory (session buffer - last 5-6 messages)
    
    Args:
        user_id: User ID (phone number)
    
    Returns:
        List of recent messages from session buffer
    """
    if not user_id:
        return []

    session = await get_user_session(user_id)
    if not session:
        return []

    buffer = session.get('session_buffer', [])
    if not isinstance(buffer, list):
        return []

    return buffer


async def should_use_long_term_memory(user_message: str, session_buffer: List[Dict[str, Any]]) -> bool:
    """
    Determine if long-term memory retrieval is needed
    Uses entity detection and better heuristics
    
    Args:
        user_message: Current user message
        session_buffer: Short-term session buffer
    
    Returns:
        bool: True if long-term memory should be retrieved
    """
    if not user_message:
        return False

    import re
    
    message_lower = user_message.lower()
    message_original = user_message.strip()

    # 1. Keywords that suggest need for long-term memory
    memory_keywords = [
        'remember', 'recall', 'before', 'earlier', 'last time', 'previously',
        'my', 'i', 'me', 'my name', 'i like', 'i prefer', 'i have',
        'my sister', 'my brother', 'my friend', 'my family',
        'where', 'when', 'who', 'what', 'how'
    ]
    has_memory_keywords = any(keyword in message_lower for keyword in memory_keywords)
    
    # 2. Entity detection - look for names, places, numbers
    # Phone numbers (7+ digits)
    phone_pattern = r'\b\d{7,}\b'
    has_phone_number = bool(re.search(phone_pattern, message_original))
    
    # Names (capitalized words, or lowercase words that might be names)
    # Look for patterns like "riya", "Riya", "what's riya", etc.
    name_patterns = [
        r'\b([A-Z][a-z]+)\b',  # Capitalized words
        r'\b([a-z]{3,})\b'     # Lowercase words (potential names)
    ]
    potential_names = []
    for pattern in name_patterns:
        matches = re.findall(pattern, message_original)
        for match in matches:
            word = (match if isinstance(match, str) else match[0] if match else '').lower()
            # Filter out common words
            common_words = {
                'the', 'what', 'where', 'when', 'who', 'how', 'can', 'you', 'tell', 'me',
                'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                'this', 'that', 'these', 'those', 'here', 'there', 'then', 'than',
                'and', 'or', 'but', 'not', 'with', 'from', 'for', 'about', 'into',
                'number', 'phone', 'contact', 'address', 'name'
            }
            if word and word not in common_words and len(word) >= 3:
                potential_names.append(word)
    
    has_potential_names = len(potential_names) > 0
    
    # Places (common place indicators)
    place_indicators = ['in', 'at', 'from', 'to', 'near', 'lives', 'located']
    has_place_context = any(indicator in message_lower for indicator in place_indicators)
    
    # 3. Question patterns that need memory
    question_patterns = [
        r'what\'?s?\s+(my|the|a|an)?\s*',  # "what's my", "what is the"
        r'who\s+(is|are|was|were)',        # "who is", "who are"
        r'where\s+(is|are|was|were|do|does|did)',  # "where is", "where do"
        r'when\s+(is|are|was|were|do|does|did)',   # "when is", "when do"
        r'how\s+(is|are|was|were|do|does|did)',    # "how is", "how do"
        r'can\s+you\s+tell\s+me',          # "can you tell me"
        r'do\s+you\s+know',                 # "do you know"
        r'remember\s+(my|the|a|an)?',      # "remember my", "remember the"
    ]
    has_question_pattern = any(re.search(pattern, message_lower) for pattern in question_patterns)
    
    # 4. Check if session buffer is empty or doesn't have enough context
    has_insufficient_context = len(session_buffer) < 2
    
    # 5. Check if query asks about specific information (numbers, names, details)
    information_queries = [
        'number', 'phone', 'contact', 'address', 'email', 'name', 'age', 'birthday',
        'location', 'place', 'city', 'country', 'address'
    ]
    has_information_query = any(query in message_lower for query in information_queries)
    
    # Decision logic: Use long-term memory if ANY of these conditions are true:
    # 1. Has memory keywords
    # 2. Has question pattern + (names/numbers/places)
    # 3. Has phone numbers (definitely need memory)
    # 4. Has potential names + question pattern
    # 5. Has information query (asking for specific data)
    # 6. Session buffer is insufficient
    
    should_use = (
        has_memory_keywords or
        has_phone_number or
        (has_question_pattern and (has_potential_names or has_place_context)) or
        (has_potential_names and has_information_query) or
        has_information_query or
        has_insufficient_context
    )
    
    if should_use:
        print(f'✅ Long-term memory retrieval triggered for: "{user_message[:50]}"')
        print(f'   Reasons: keywords={has_memory_keywords}, phone={has_phone_number}, names={has_potential_names}, question={has_question_pattern}, info_query={has_information_query}')
    
    return should_use


async def retrieve_long_term_memories(
    user_id: str,
    user_message: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant long-term memories using vector search
    
    Args:
        user_id: User ID (phone number)
        user_message: User message to search for
        limit: Maximum number of memories to retrieve
    
    Returns:
        List of relevant memories
    """
    if not user_id or not user_message:
        return []

    try:
        # Generate query embedding with timeout
        try:
            query_embedding = await asyncio.wait_for(
                generate_query_embedding(user_message),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            print('⚠️  Embedding generation timed out')
            return []
        except Exception as error:
            print(f'⚠️  Failed to generate query embedding: {error}')
            return []

        if not query_embedding:
            print('⚠️  Empty embedding returned')
            return []

        # Use hybrid search (semantic + keyword) for better results
        try:
            memories = await asyncio.wait_for(
                retrieve_memories_hybrid(
                    user_id=user_id,
                    user_message=user_message,
                    query_embedding=query_embedding,
                    limit=limit,
                    similarity_threshold=0.5
                ),
                timeout=5.0  # Slightly longer timeout for hybrid search
            )
            
            # Log results for debugging
            if memories:
                scores = [m.get('similarity', 0) for m in memories]
                print(f'🔍 Hybrid search: {len(memories)} results, scores: {[f"{s:.3f}" for s in scores[:5]]}')
            else:
                print(f'⚠️  Hybrid search returned no results for: "{user_message[:50]}"')
            
            # If hybrid search returns few results, try adaptive threshold fallback
            if len(memories) < 3:
                print('⚠️  Hybrid search returned few results, trying adaptive threshold...')
                thresholds = [0.4, 0.3, 0.0]  # 0.0 means no threshold (return top N)
                
                for threshold in thresholds:
                    try:
                        retrieved = await asyncio.wait_for(
                            retrieve_memories(
                                user_id=user_id,
                                query_embedding=query_embedding,
                                limit=limit * 2,
                                similarity_threshold=threshold
                            ),
                            timeout=3.0
                        )
                        
                        if retrieved:
                            scores = [m.get('similarity', 0) for m in retrieved]
                            print(f'🔍 Fallback retrieval (threshold={threshold}): {len(retrieved)} results, scores: {[f"{s:.3f}" for s in scores[:5]]}')
                        
                        if len(retrieved) >= 3:
                            memories = retrieved[:limit]
                            break
                        elif threshold == 0.0:
                            memories = retrieved[:limit]
                            if memories:
                                print(f'⚠️  Using top {len(memories)} results below normal threshold')
                            break
                        else:
                            memories = retrieved
                            
                    except asyncio.TimeoutError:
                        print(f'⚠️  Fallback retrieval timed out at threshold {threshold}')
                        if threshold == 0.0:
                            break
                        continue
                    except Exception as error:
                        print(f'⚠️  Error in fallback search at threshold {threshold}: {error}')
                        if threshold == 0.0:
                            break
                        continue
                        
        except asyncio.TimeoutError:
            print('⚠️  Hybrid memory retrieval timed out')
            return []
        except Exception as error:
            print(f'❌ Error in hybrid memory retrieval: {error}')
            return []

        return memories
    except Exception as error:
        print(f'❌ Error retrieving long-term memories: {error}')
        return []


async def extract_memory_facts(
    user_message: str,
    session_buffer: List[Dict[str, Any]],
    ai_response: str
) -> List[Dict[str, Any]]:
    """
    Extract memory facts from conversation using Gemini
    
    Args:
        user_message: User's message
        session_buffer: Session buffer context
        ai_response: AI's response
    
    Returns:
        List of extracted facts
    """
    try:
        from ai import extract_memory_facts as ai_extract_facts
        return await ai_extract_facts(user_message, session_buffer, ai_response)
    except Exception as error:
        print(f'❌ Error extracting memory facts: {error}')
        return []


async def save_extracted_memories(
    user_id: str,
    facts: List[Dict[str, Any]]
) -> int:
    """
    Save extracted memory facts to database
    
    Args:
        user_id: User ID (phone number)
        facts: List of fact dictionaries with content, type, metadata
    
    Returns:
        int: Number of memories successfully saved
    """
    if not user_id or not facts:
        return 0

    saved_count = 0
    
    for fact in facts:
        if not fact or not isinstance(fact, dict):
            continue

        content = fact.get('content', '').strip()
        if not content:
            continue

        try:
            # Generate embedding for memory with timeout
            embedding = None
            try:
                embedding = await asyncio.wait_for(
                    generate_embedding(content),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                print(f'⚠️  Embedding generation timed out for: {content[:50]}')
            except Exception as error:
                print(f'⚠️  Error generating embedding: {error}')
            
            # Prepare metadata
            metadata = {
                'type': fact.get('type', 'general'),
                'tags': fact.get('tags', []),
                'extracted_at': datetime.now().isoformat()
            }

            # Save memory (even without embedding as fallback)
            memory_id = await save_memory(
                user_id=user_id,
                content=content,
                embedding=embedding,
                metadata=metadata
            )

            if memory_id:
                saved_count += 1
        except Exception as error:
            print(f'⚠️  Error saving memory fact: {error}')
            continue

    return saved_count


async def process_session_message(
    user_id: str,
    user_message: str,
    ai_response: str
) -> Dict[str, Any]:
    """
    Process a message in the session context
    
    Args:
        user_id: User ID (phone number)
        user_message: User's message
        ai_response: AI's response
    
    Returns:
        Dict with session info and memory retrieval results
    """
    # Default return value in case of errors
    default_result = {
        'session': None,
        'short_term_memory': [],
        'long_term_memories': [],
        'use_long_term': False
    }

    try:
        # Check session status
        session_status = await check_session_status(user_id)
        
        # Finalize old session if needed
        if session_status['needs_finalization']:
            try:
                # Generate batch summary before finalizing
                session_buffer = session_status['session'].get('session_buffer', []) if session_status['session'] else []
                if session_buffer:
                    try:
                        from ai import generate_batch_memory_summary
                        batch_summary = await generate_batch_memory_summary(session_buffer)
                        if batch_summary.get('facts'):
                            await save_extracted_memories(user_id, batch_summary['facts'])
                    except Exception as error:
                        print(f'⚠️  Error generating batch summary: {error}')
                
                await finalize_session(user_id)
                # Create new session
                session_status = await check_session_status(user_id)
            except Exception as error:
                print(f'⚠️  Error finalizing session: {error}')

        # Get short-term memory
        try:
            short_term = await get_short_term_memory(user_id)
        except Exception as error:
            print(f'⚠️  Error getting short-term memory: {error}')
            short_term = []
        
        # Determine if long-term memory is needed
        use_long_term = False
        try:
            use_long_term = await should_use_long_term_memory(user_message, short_term)
        except Exception as error:
            print(f'⚠️  Error determining long-term memory need: {error}')
        
        # Retrieve long-term memories if needed
        long_term_memories = []
        if use_long_term:
            try:
                long_term_memories = await retrieve_long_term_memories(user_id, user_message)
            except Exception as error:
                print(f'⚠️  Error retrieving long-term memories: {error}')
                long_term_memories = []

        # Update session with new message
        try:
            await update_session(user_id, {
                'userMessage': user_message,
                'aiResponse': ai_response,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as error:
            print(f'⚠️  Error updating session: {error}')

        return {
            'session': session_status.get('session'),
            'short_term_memory': short_term,
            'long_term_memories': long_term_memories,
            'use_long_term': use_long_term
        }
    except Exception as error:
        print(f'❌ Error processing session message: {error}')
        return default_result

