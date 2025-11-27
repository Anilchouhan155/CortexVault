#!/usr/bin/env python3
"""
Memory Brain System Test Suite
Tests the complete memory system including:
- Session management (1-hour window)
- Short-term memory (session buffer)
- Long-term memory retrieval (vector search)
- Memory extraction
- Batch memory updates
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Import modules
try:
    from database import (
        get_or_create_user,
        save_memory,
        retrieve_memories,
        get_user_session,
        update_session,
        finalize_session
    )
    from services.embedding_service import generate_embedding, generate_query_embedding
    from services.memory_service import (
        check_session_status,
        get_short_term_memory,
        should_use_long_term_memory,
        retrieve_long_term_memories,
        extract_memory_facts,
        save_extracted_memories,
        process_session_message
    )
    from ai import (
        generate_ai_response,
        extract_memory_facts as ai_extract_facts,
        generate_batch_memory_summary
    )
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

# Colors for terminal output
class Colors:
    RESET = '\033[0m'
    GREEN = '\033[32m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    MAGENTA = '\033[35m'

# Test results
passed_tests = 0
failed_tests = 0
test_results: List[Dict[str, Any]] = []

def log(message: str, color: str = Colors.RESET) -> None:
    """Print colored message"""
    print(f"{color}{message}{Colors.RESET}")

def log_test(name: str, status: str, message: str = '', details: Any = None) -> None:
    """Log test result"""
    icon = '✅' if status == 'PASS' else '❌' if status == 'FAIL' else '⚠️'
    color = Colors.GREEN if status == 'PASS' else Colors.RED if status == 'FAIL' else Colors.YELLOW
    log(f"{icon} {name}{': ' + message if message else ''}", color)
    
    if details:
        log(f"   Details: {details}", Colors.BLUE)
    
    test_results.append({'name': name, 'status': status, 'message': message, 'details': details})
    global passed_tests, failed_tests
    if status == 'PASS':
        passed_tests += 1
    elif status == 'FAIL':
        failed_tests += 1

def log_section(title: str) -> None:
    """Log section header"""
    print()
    log('=' * 70, Colors.CYAN)
    log(f"  {title}", Colors.CYAN)
    log('=' * 70, Colors.CYAN)
    print()

def log_subsection(title: str) -> None:
    """Log subsection header"""
    print()
    log(f"--- {title} ---", Colors.MAGENTA)


# ============================================
# DUMMY USER DATA FOR TESTING
# ============================================

DUMMY_USER_PHONE = "919999999999"
DUMMY_USER_NAME = "TestUser"

# Sample conversation messages to simulate a real conversation
DUMMY_CONVERSATIONS = [
    {
        'userMessage': "Hi! My name is Alex and I love playing guitar.",
        'aiResponse': "Hey Alex! That's awesome! What kind of music do you like to play?",
        'expected_facts': [
            {'type': 'person', 'content': 'User name is Alex'},
            {'type': 'preference', 'content': 'User loves playing guitar'}
        ]
    },
    {
        'userMessage': "I mostly play rock and jazz. My sister Sarah lives in Mumbai.",
        'aiResponse': "Rock and jazz are great! How long has Sarah been in Mumbai?",
        'expected_facts': [
            {'type': 'preference', 'content': 'User plays rock and jazz music'},
            {'type': 'person', 'content': 'User has a sister named Sarah'},
            {'type': 'place', 'content': 'Sarah lives in Mumbai'}
        ]
    },
    {
        'userMessage': "She moved there 2 years ago. I work as a software engineer.",
        'aiResponse': "That's interesting! What kind of software do you work on?",
        'expected_facts': [
            {'type': 'detail', 'content': 'Sarah moved to Mumbai 2 years ago'},
            {'type': 'detail', 'content': 'User works as a software engineer'}
        ]
    },
    {
        'userMessage': "I build web applications. My favorite programming language is Python.",
        'aiResponse': "Python is great! What frameworks do you use?",
        'expected_facts': [
            {'type': 'detail', 'content': 'User builds web applications'},
            {'type': 'preference', 'content': 'User favorite programming language is Python'}
        ]
    },
    {
        'userMessage': "I use FastAPI and React. I have a dog named Max.",
        'aiResponse': "FastAPI and React are a solid stack! How old is Max?",
        'expected_facts': [
            {'type': 'preference', 'content': 'User uses FastAPI and React'},
            {'type': 'person', 'content': 'User has a dog named Max'}
        ]
    },
    {
        'userMessage': "Max is 3 years old. I'm planning a trip to Goa next month.",
        'aiResponse': "Goa sounds fun! What are you planning to do there?",
        'expected_facts': [
            {'type': 'detail', 'content': 'Max is 3 years old'},
            {'type': 'event', 'content': 'User planning trip to Goa next month'}
        ]
    }
]


# ============================================
# TEST FUNCTIONS
# ============================================

async def test_environment_setup() -> bool:
    """Test environment and dependencies"""
    log_section('1. Environment & Dependencies Check')
    
    required_vars = [
        'GEMINI_API_KEY',
        'SUPABASE_URL',
        'SUPABASE_SERVICE_ROLE_KEY'
    ]
    
    all_present = True
    for var_name in required_vars:
        if os.getenv(var_name):
            log_test(f'Env var: {var_name}', 'PASS')
        else:
            log_test(f'Env var: {var_name}', 'FAIL', 'Missing')
            all_present = False
    
    return all_present


async def test_user_creation() -> Optional[str]:
    """Test user creation and return user_id"""
    log_section('2. User Creation Test')
    
    try:
        user_id = await get_or_create_user(DUMMY_USER_PHONE, DUMMY_USER_NAME)
        
        if user_id:
            log_test('User creation', 'PASS', f'User ID: {user_id}')
            return user_id
        else:
            log_test('User creation', 'FAIL', 'Failed to create user')
            return None
    except Exception as error:
        log_test('User creation', 'FAIL', str(error))
        return None


async def test_embedding_generation() -> bool:
    """Test embedding generation"""
    log_section('3. Embedding Generation Test')
    
    test_text = "My name is Alex and I love playing guitar."
    
    try:
        # Test document embedding
        embedding = await generate_embedding(test_text)
        if embedding and len(embedding) > 0:
            log_test('Document embedding generation', 'PASS', f'Dimension: {len(embedding)}')
            if len(embedding) == 768:
                log_test('Embedding dimension', 'PASS', 'Correct dimension (768)')
            else:
                log_test('Embedding dimension', 'FAIL', f'Expected 768, got {len(embedding)}')
        else:
            log_test('Document embedding generation', 'FAIL', 'Empty embedding')
            return False
        
        # Test query embedding
        query_embedding = await generate_query_embedding(test_text)
        if query_embedding and len(query_embedding) > 0:
            log_test('Query embedding generation', 'PASS', f'Dimension: {len(query_embedding)}')
        else:
            log_test('Query embedding generation', 'FAIL', 'Empty embedding')
            return False
        
        return True
    except Exception as error:
        log_test('Embedding generation', 'FAIL', str(error))
        return False


async def test_memory_storage(user_id: str) -> List[str]:
    """Test storing memories with embeddings"""
    log_section('4. Memory Storage Test')
    
    test_memories = [
        {
            'content': 'User name is Alex',
            'type': 'person',
            'tags': ['name', 'identity']
        },
        {
            'content': 'User loves playing guitar',
            'type': 'preference',
            'tags': ['music', 'hobby']
        },
        {
            'content': 'User has a sister named Sarah',
            'type': 'person',
            'tags': ['family', 'sister']
        },
        {
            'content': 'Sarah lives in Mumbai',
            'type': 'place',
            'tags': ['location', 'family']
        },
        {
            'content': 'User works as a software engineer',
            'type': 'detail',
            'tags': ['work', 'profession']
        }
    ]
    
    saved_memory_ids: List[str] = []
    
    for memory in test_memories:
        try:
            # Generate embedding
            embedding = await generate_embedding(memory['content'])
            
            # Save memory
            memory_id = await save_memory(
                user_id=user_id,
                content=memory['content'],
                embedding=embedding,
                metadata={
                    'type': memory['type'],
                    'tags': memory['tags']
                }
            )
            
            if memory_id:
                saved_memory_ids.append(memory_id)
                log_test(f'Memory storage: {memory["content"][:30]}...', 'PASS', f'ID: {memory_id}')
            else:
                log_test(f'Memory storage: {memory["content"][:30]}...', 'FAIL', 'Failed to save')
        except Exception as error:
            log_test(f'Memory storage: {memory["content"][:30]}...', 'FAIL', str(error))
    
    return saved_memory_ids


async def test_memory_retrieval(user_id: str) -> bool:
    """Test memory retrieval via vector search"""
    log_section('5. Memory Retrieval Test')
    
    test_queries = [
        {
            'query': "What's my name?",
            'expected_content': 'Alex'
        },
        {
            'query': "What do I like?",
            'expected_content': 'guitar'
        },
        {
            'query': "Where does my sister live?",
            'expected_content': 'Mumbai'
        },
        {
            'query': "What's my job?",
            'expected_content': 'software engineer'
        }
    ]
    
    all_passed = True
    
    for test_query in test_queries:
        try:
            # Generate query embedding
            query_embedding = await generate_query_embedding(test_query['query'])
            if not query_embedding:
                log_test(f'Retrieval: {test_query["query"]}', 'FAIL', 'Failed to generate query embedding')
                all_passed = False
                continue
            
            # Retrieve memories
            memories = await retrieve_memories(
                user_id=user_id,
                query_embedding=query_embedding,
                limit=5,
                similarity_threshold=0.5  # Lower threshold for testing
            )
            
            if memories and len(memories) > 0:
                # Check if expected content is in retrieved memories
                found = False
                for mem in memories:
                    if test_query['expected_content'].lower() in mem.get('content', '').lower():
                        found = True
                        log_test(
                            f'Retrieval: {test_query["query"]}',
                            'PASS',
                            f'Found: {mem.get("content", "")[:50]}... (similarity: {mem.get("similarity", 0):.3f})'
                        )
                        break
                
                if not found:
                    log_test(
                        f'Retrieval: {test_query["query"]}',
                        'FAIL',
                        f'Expected "{test_query["expected_content"]}" not found in results'
                    )
                    log(f"   Retrieved: {[m.get('content', '')[:30] for m in memories]}", Colors.YELLOW)
                    all_passed = False
            else:
                log_test(f'Retrieval: {test_query["query"]}', 'FAIL', 'No memories retrieved')
                all_passed = False
        except Exception as error:
            log_test(f'Retrieval: {test_query["query"]}', 'FAIL', str(error))
            all_passed = False
    
    return all_passed


async def test_session_management(user_id: str) -> bool:
    """Test session management (1-hour window)"""
    log_section('6. Session Management Test')
    
    try:
        # Test 1: Create new session
        session = await get_user_session(user_id)
        if session:
            log_test('Session creation', 'PASS', f'Session ID: {session.get("id", "unknown")}')
        else:
            log_test('Session creation', 'FAIL', 'Failed to create session')
            return False
        
        # Test 2: Check active session status
        status = await check_session_status(user_id)
        if status['session_status'] == 'active':
            log_test('Active session check', 'PASS', 'Session is active')
        else:
            log_test('Active session check', 'FAIL', f'Unexpected status: {status["session_status"]}')
            return False
        
        # Test 3: Update session with messages
        for i, conv in enumerate(DUMMY_CONVERSATIONS[:3]):
            success = await update_session(user_id, {
                'userMessage': conv['userMessage'],
                'aiResponse': conv['aiResponse'],
                'timestamp': datetime.now().isoformat()
            })
            if success:
                log_test(f'Session update {i+1}', 'PASS', f'Added: {conv["userMessage"][:30]}...')
            else:
                log_test(f'Session update {i+1}', 'FAIL', 'Failed to update session')
        
        # Test 4: Get short-term memory
        short_term = await get_short_term_memory(user_id)
        if short_term and len(short_term) > 0:
            log_test('Short-term memory retrieval', 'PASS', f'Retrieved {len(short_term)} messages')
            log(f"   Messages: {[m.get('userMessage', '')[:30] for m in short_term]}", Colors.BLUE)
        else:
            log_test('Short-term memory retrieval', 'FAIL', 'No messages in buffer')
        
        # Test 5: Check if long-term memory is needed
        use_long_term = await should_use_long_term_memory(
            "What's my name?",
            short_term
        )
        if use_long_term:
            log_test('Long-term memory decision', 'PASS', 'Correctly identified need for long-term memory')
        else:
            log_test('Long-term memory decision', 'FAIL', 'Should use long-term memory for name query')
        
        return True
    except Exception as error:
        log_test('Session management', 'FAIL', str(error))
        return False


async def test_memory_extraction(user_id: str) -> bool:
    """Test memory extraction from conversations"""
    log_section('7. Memory Extraction Test')
    
    all_passed = True
    
    for i, conv in enumerate(DUMMY_CONVERSATIONS[:3]):
        try:
            # Get current session buffer
            short_term = await get_short_term_memory(user_id)
            
            # Extract facts
            facts = await extract_memory_facts(
                user_message=conv['userMessage'],
                session_buffer=short_term,
                ai_response=conv['aiResponse']
            )
            
            if facts and len(facts) > 0:
                log_test(
                    f'Extraction {i+1}: {conv["userMessage"][:30]}...',
                    'PASS',
                    f'Extracted {len(facts)} fact(s)'
                )
                for fact in facts:
                    log(f"   - {fact.get('type', 'unknown')}: {fact.get('content', '')[:50]}", Colors.BLUE)
                
                # Save extracted memories
                saved_count = await save_extracted_memories(user_id, facts)
                if saved_count > 0:
                    log_test(f'Memory save {i+1}', 'PASS', f'Saved {saved_count} memory(ies)')
                else:
                    log_test(f'Memory save {i+1}', 'FAIL', 'Failed to save memories')
                    all_passed = False
            else:
                log_test(
                    f'Extraction {i+1}: {conv["userMessage"][:30]}...',
                    'FAIL',
                    'No facts extracted'
                )
                all_passed = False
        except Exception as error:
            log_test(f'Extraction {i+1}', 'FAIL', str(error))
            all_passed = False
    
    return all_passed


async def test_end_to_end_conversation(user_id: str) -> bool:
    """Test complete end-to-end conversation flow"""
    log_section('8. End-to-End Conversation Test')
    
    try:
        # Simulate a conversation flow
        conversation_flow = [
            "Hi! My name is Alex.",
            "I love playing guitar and my favorite band is The Beatles.",
            "What's my name?",
            "Where do I live?",
            "I live in Bangalore."
        ]
        
        for i, user_msg in enumerate(conversation_flow):
            log_subsection(f'Message {i+1}: {user_msg}')
            
            # Process session message
            memory_context = await process_session_message(
                user_id=user_id,
                user_message=user_msg,
                ai_response=f"AI response to: {user_msg}"  # Dummy response
            )
            
            # Get memories
            short_term = memory_context.get('short_term_memory', [])
            long_term = memory_context.get('long_term_memories', [])
            
            log(f"Short-term memory: {len(short_term)} messages", Colors.BLUE)
            log(f"Long-term memories: {len(long_term)} retrieved", Colors.BLUE)
            
            if long_term:
                for mem in long_term:
                    log(f"  - {mem.get('content', '')[:50]} (similarity: {mem.get('similarity', 0):.3f})", Colors.CYAN)
            
            # Generate AI response with memory context
            try:
                ai_response = await generate_ai_response(
                    user_message=user_msg,
                    conversation_history=short_term,
                    user_name=DUMMY_USER_NAME,
                    target_language='en',
                    retrieved_memories=long_term
                )
                log(f"AI Response: {ai_response[:100]}", Colors.GREEN)
            except Exception as error:
                log(f"AI Response generation failed: {error}", Colors.YELLOW)
            
            # Extract and save memories
            try:
                facts = await extract_memory_facts(
                    user_message=user_msg,
                    session_buffer=short_term,
                    ai_response=ai_response if 'ai_response' in locals() else "Dummy response"
                )
                if facts:
                    saved = await save_extracted_memories(user_id, facts)
                    log(f"Extracted and saved {saved} memory(ies)", Colors.MAGENTA)
            except Exception as error:
                log(f"Memory extraction failed: {error}", Colors.YELLOW)
        
        log_test('End-to-end conversation', 'PASS', 'Completed full conversation flow')
        return True
    except Exception as error:
        log_test('End-to-end conversation', 'FAIL', str(error))
        return False


async def test_batch_memory_update(user_id: str) -> bool:
    """Test batch memory update on session end"""
    log_section('9. Batch Memory Update Test')
    
    try:
        # Get current session buffer
        short_term = await get_short_term_memory(user_id)
        
        if not short_term or len(short_term) == 0:
            log_test('Batch update', 'FAIL', 'No session buffer to summarize')
            return False
        
        # Generate batch summary
        batch_summary = await generate_batch_memory_summary(short_term)
        
        if batch_summary:
            summary = batch_summary.get('summary', '')
            facts = batch_summary.get('facts', [])
            
            if summary:
                log_test('Batch summary generation', 'PASS', f'Summary: {summary[:100]}...')
            else:
                log_test('Batch summary generation', 'FAIL', 'Empty summary')
            
            if facts and len(facts) > 0:
                log_test('Batch facts extraction', 'PASS', f'Extracted {len(facts)} fact(s)')
                for fact in facts:
                    log(f"   - {fact.get('type', 'unknown')}: {fact.get('content', '')[:50]}", Colors.BLUE)
                
                # Save batch memories
                saved_count = await save_extracted_memories(user_id, facts)
                if saved_count > 0:
                    log_test('Batch memory save', 'PASS', f'Saved {saved_count} memory(ies)')
                else:
                    log_test('Batch memory save', 'FAIL', 'Failed to save batch memories')
            else:
                log_test('Batch facts extraction', 'FAIL', 'No facts extracted')
        else:
            log_test('Batch summary', 'FAIL', 'Failed to generate batch summary')
            return False
        
        return True
    except Exception as error:
        log_test('Batch memory update', 'FAIL', str(error))
        return False


async def test_session_finalization(user_id: str) -> bool:
    """Test session finalization"""
    log_section('10. Session Finalization Test')
    
    try:
        # Finalize session
        success = await finalize_session(user_id)
        if success:
            log_test('Session finalization', 'PASS', 'Session finalized successfully')
        else:
            log_test('Session finalization', 'FAIL', 'Failed to finalize session')
            return False
        
        # Check that buffer is cleared
        short_term = await get_short_term_memory(user_id)
        if not short_term or len(short_term) == 0:
            log_test('Session buffer cleared', 'PASS', 'Buffer is empty after finalization')
        else:
            log_test('Session buffer cleared', 'FAIL', f'Buffer still has {len(short_term)} messages')
        
        return True
    except Exception as error:
        log_test('Session finalization', 'FAIL', str(error))
        return False


# ============================================
# MAIN TEST RUNNER
# ============================================

async def run_all_tests() -> int:
    """Run all memory brain tests"""
    print()
    log('🧠 Memory Brain System Test Suite', Colors.CYAN)
    log('=' * 70, Colors.CYAN)
    print()
    
    results: List[bool] = []
    user_id: Optional[str] = None
    
    # Test 1: Environment
    env_ok = await test_environment_setup()
    results.append(env_ok)
    if not env_ok:
        log('\n⚠️  Environment setup failed. Some tests may fail.', Colors.YELLOW)
    
    # Test 2: User creation
    user_id = await test_user_creation()
    if not user_id:
        log('\n❌ User creation failed. Cannot continue with memory tests.', Colors.RED)
        return 1
    results.append(True)
    
    # Test 3: Embedding generation
    embedding_ok = await test_embedding_generation()
    results.append(embedding_ok)
    
    # Test 4: Memory storage
    memory_ids = await test_memory_storage(user_id)
    results.append(len(memory_ids) > 0)
    
    # Test 5: Memory retrieval
    retrieval_ok = await test_memory_retrieval(user_id)
    results.append(retrieval_ok)
    
    # Test 6: Session management
    session_ok = await test_session_management(user_id)
    results.append(session_ok)
    
    # Test 7: Memory extraction
    extraction_ok = await test_memory_extraction(user_id)
    results.append(extraction_ok)
    
    # Test 8: End-to-end conversation
    e2e_ok = await test_end_to_end_conversation(user_id)
    results.append(e2e_ok)
    
    # Test 9: Batch memory update
    batch_ok = await test_batch_memory_update(user_id)
    results.append(batch_ok)
    
    # Test 10: Session finalization
    finalize_ok = await test_session_finalization(user_id)
    results.append(finalize_ok)
    
    # Summary
    print()
    log('=' * 70, Colors.CYAN)
    log('📊 Test Summary', Colors.CYAN)
    log('=' * 70, Colors.CYAN)
    print()
    log(f'✅ Passed: {passed_tests}', Colors.GREEN)
    log(f'❌ Failed: {failed_tests}', Colors.RED)
    log(f'📈 Total: {passed_tests + failed_tests}', Colors.BLUE)
    print()
    
    # Detailed results
    log('Detailed Results:', Colors.CYAN)
    for result in test_results:
        status_icon = '✅' if result['status'] == 'PASS' else '❌'
        log(f"{status_icon} {result['name']}: {result['status']}", 
            Colors.GREEN if result['status'] == 'PASS' else Colors.RED)
        if result.get('message'):
            log(f"   {result['message']}", Colors.BLUE)
    print()
    
    if all(results):
        log('✅ ALL TESTS PASSED!', Colors.GREEN)
        return 0
    else:
        log('⚠️  SOME TESTS FAILED - Review details above', Colors.YELLOW)
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

