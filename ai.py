"""
AI Integration Module
Handles Google Gemini AI responses with friend-first personality
"""

import os
import re
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Gemini AI
genai_model: Optional[Any] = None


def initialize_gemini() -> bool:
    """
    Initialize Gemini AI client
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global genai_model
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print('❌ GEMINI_API_KEY not configured')
        return False

    try:
        genai.configure(api_key=api_key)
        # Use gemini-2.5-flash-lite (fast and capable model)
        genai_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        print('✅ Gemini AI initialized successfully')
        return True
    except Exception as error:
        print(f'❌ Error initializing Gemini: {error}')
        return False

# Initialize on module load
if not initialize_gemini():
    print('⚠️  Gemini AI not initialized. AI responses will fail.')


async def generate_ai_response(
    user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    user_name: str = 'User',
    target_language: str = 'en',
    retrieved_memories: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generate AI response with personality and memory context
    
    Args:
        user_message: User's message text
        conversation_history: List of previous conversations (short-term)
        user_name: Name of the user
        target_language: Language code ('en', 'hi', 'auto')
        retrieved_memories: List of relevant long-term memories
    
    Returns:
        str: AI response text
    """
    # Re-initialize if model is null
    if not genai_model and not initialize_gemini():
        print('⚠️  Gemini AI not configured')
        return "Hey! Something went wrong on my end. Can you try again? 😅"

    if not user_message or not user_message.strip():
        return "Hey! I didn't catch that. Can you send that again?"

    try:
        # Build context from recent conversations (short-term)
        short_term_context = ''
        if conversation_history:
            short_term_context = '\n\n'.join([
                f"{user_name}: {conv.get('userMessage', conv.get('user_message', ''))}\n"
                f"You: {conv.get('aiResponse', conv.get('ai_response', ''))}"
                for conv in conversation_history
            ])

        # Build long-term memory context with emphasis
        memory_context = ''
        if retrieved_memories and len(retrieved_memories) > 0:
            memory_texts = [mem.get('content', '') for mem in retrieved_memories if mem.get('content')]
            if memory_texts:
                # Format memories prominently with similarity scores for important context
                # Also make memories more explicit based on query context
                memory_list = []
                query_lower = user_message.lower()
                
                for i, mem in enumerate(retrieved_memories[:5], 1):  # Top 5 memories
                    content = mem.get('content', '')
                    similarity = mem.get('similarity', 0.0)
                    if content:
                        # Make memory more explicit if query is asking about "my" or user's attributes
                        formatted_content = content
                        if 'my' in query_lower or 'i' in query_lower.split():
                            # If memory is about user attribute, make it explicit
                            if 'height' in content.lower() and 'height' in query_lower:
                                formatted_content = f"Your height is {content.split('is')[-1].strip()}" if 'is' in content else content
                            elif 'name' in content.lower() and 'name' in query_lower:
                                formatted_content = f"Your name is {content.split('is')[-1].strip()}" if 'is' in content else content
                            elif 'phone' in content.lower() or 'number' in content.lower():
                                # Keep as is for phone numbers
                                formatted_content = content
                        
                        # Include similarity score for high-confidence memories
                        if similarity >= 0.6:
                            memory_list.append(f"{i}. {formatted_content} [High confidence: {similarity:.2f}]")
                        else:
                            memory_list.append(f"{i}. {formatted_content}")
                
                memory_context = '\n'.join(memory_list)
                
                # Log what memories are being sent to AI for debugging
                print(f'📝 Sending {len(retrieved_memories)} memories to AI:')
                for mem in retrieved_memories[:5]:
                    print(f'   - {mem.get("content", "")[:60]} (similarity: {mem.get("similarity", 0):.3f})')

        # Language-specific instructions
        language_instructions = {
            'hi': 'Respond in Hindi (हिंदी). Use natural, conversational Hindi. Keep it casual and friendly.',
            'en': 'Respond in English. Keep it casual and friendly.',
            'auto': 'Respond in the same language as the user\'s message. Match their language naturally.'
        }

        lang_instruction = language_instructions.get(target_language, language_instructions['auto'])

        # Build context sections
        short_term_section = ''
        if short_term_context:
            short_term_section = f'Short-Term Messages:\n\n{short_term_context}\n\n'

        memory_section = ''
        if memory_context:
            # Check if query is asking about something in memories
            query_lower = user_message.lower()
            has_relevant_memory = False
            relevant_memory_content = None
            
            for mem in retrieved_memories[:5]:
                mem_content = mem.get('content', '').lower()
                # Check if memory is relevant to query
                if any(word in mem_content for word in query_lower.split() if len(word) > 3):
                    has_relevant_memory = True
                    relevant_memory_content = mem.get('content', '')
                    break
            
            memory_section = f"""CRITICAL - Long-Term Memories (YOU HAVE THIS INFORMATION):

{memory_context}

MANDATORY RULES - READ CAREFULLY:
1. The memories above contain FACTS about the user that you KNOW
2. If the user asks "what's my height" and memory says "height is 5'11"", you MUST respond with "Your height is 5'11"" or "You're 5'11""
3. If the user asks "what's [name]'s number" and memory contains that number, you MUST provide it
4. NEVER say "I don't know", "I don't remember", "I'm not sure", or "I don't have that" if the information is in the memories above
5. The memories are YOUR knowledge - treat them as facts you know about the user
6. For questions matching memory content, use the memory information DIRECTLY
7. Be confident and direct - if you have the information, share it immediately
8. Example: User asks "what's my height?" → Memory: "height is 5'11"" → Response: "You're 5'11"!" (NOT "I don't remember")
9. Example: User asks "what's Riya's number?" → Memory: "Riya's phone number is 8750227749" → Response: "Riya's number is 8750227749" (NOT "I'm not sure")

REMEMBER: These memories are YOUR knowledge. Use them confidently.

"""
        
        system_prompt = f"""You are NexaBrain, a fast conversational AI friend. You:
- Talk casually like a close friend
- Never use "please" or formal language  
- Remember previous conversations and long-term memories
- Show genuine interest in the user's life
- Are supportive and caring
- Keep responses under 2 sentences for WhatsApp
- Use the user's name ({user_name}) occasionally but naturally
- Be conversational and warm, not robotic
- {lang_instruction}

{short_term_section}{memory_section}User's message: {user_message}

IMPORTANT: Before responding, check if the user's question matches any information in the memories above. If it does, use that information directly in your response. Do not say you don't know if the information is in the memories.

Respond naturally as their AI friend in the same language:"""

        # Generate response using Gemini (run in executor to make it async)
        start_time = datetime.now()
        
        # Run synchronous Gemini call in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai_model.generate_content(system_prompt)
        )
        
        api_duration = (datetime.now() - start_time).total_seconds() * 1000
        print(f'🤖 Gemini API call completed in {api_duration:.0f}ms')
        
        text = result.text

        if not text or not text.strip():
            raise Exception('Empty response from Gemini')

        # Ensure response follows friend-first rules
        return clean_friend_response(text.strip())
        
    except Exception as error:
        print(f'❌ Error generating AI response: {error}')
        
        # Provide helpful error messages
        error_msg = str(error)
        if 'API_KEY' in error_msg:
            print('⚠️  Invalid or missing GEMINI_API_KEY')
        elif 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
            print('⚠️  Gemini API quota/rate limit exceeded')
        elif 'timeout' in error_msg.lower():
            print('⚠️  Gemini API request timed out - API may be slow or unresponsive')
        
        # Fallback response if AI fails
        return "Hey! Something went wrong on my end. Can you try again? 😅"


def clean_friend_response(response: str) -> str:
    """
    Clean response to ensure friend-first personality
    
    Args:
        response: Raw AI response
    
    Returns:
        str: Cleaned response
    """
    if not response or not response.strip():
        return response
    
    # Remove formal language
    cleaned = response
    cleaned = re.sub(r'please', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'sir', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'madam', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bI\'d be happy to\b', "I'd love to", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bHow may I\b', "How can I", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bIs there anything else\b', "Anything else", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bI am\b', "I'm", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bI will\b', "I'll", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bI would\b', "I'd", cleaned, flags=re.IGNORECASE)
    
    # Ensure it starts casually
    if cleaned.startswith('I am') or cleaned.startswith('I will'):
        cleaned = re.sub(r'^I am', "I'm", cleaned)
        cleaned = re.sub(r'^I will', "I'll", cleaned)
    
    # Remove excessive formality markers
    cleaned = re.sub(r'^Certainly[,!]?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^Of course[,!]?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^Absolutely[,!]?\s*', '', cleaned, flags=re.IGNORECASE)
    
    # Trim and ensure it's not too long (WhatsApp limit is 4096, but we want shorter)
    cleaned = cleaned.strip()
    
    # If response is too long, truncate intelligently
    if len(cleaned) > 500:
        # Try to truncate at sentence boundary
        sentences = re.findall(r'[^.!?]+[.!?]+', cleaned)
        if sentences:
            truncated = ''
            for sentence in sentences:
                if len(truncated + sentence) <= 500:
                    truncated += sentence
                else:
                    break
            cleaned = truncated or cleaned[:497] + '...'
        else:
            cleaned = cleaned[:497] + '...'
    
    return cleaned.strip()


async def extract_memory_facts(
    user_message: str,
    session_buffer: List[Dict[str, Any]],
    ai_response: str
) -> List[Dict[str, Any]]:
    """
    Extract long-term relevant facts from conversation using Gemini
    
    Args:
        user_message: User's message
        session_buffer: Session buffer (last 5-6 messages)
        ai_response: AI's response
    
    Returns:
        List of extracted facts with type, content, and tags
    """
    if not genai_model and not initialize_gemini():
        print('⚠️  Gemini AI not configured for memory extraction')
        return []

    if not user_message or not user_message.strip():
        return []

    try:
        # Build session context
        session_context = ''
        if session_buffer:
            session_context = '\n'.join([
                f"User: {msg.get('userMessage', '')}\nAI: {msg.get('aiResponse', '')}"
                for msg in session_buffer[-3:]  # Last 3 messages for context
            ])

        # Memory extraction prompt
        extraction_prompt = f"""From the message below, extract ONLY long-term relevant facts.

Return JSON in this structure:
{{
  "facts": [
    {{ "type": "person|place|preference|event|detail", "value": "fact content", "tags": [] }}
  ]
}}

Rules:
- Extract stable facts about the user (names, preferences, locations, relationships)
- Ignore temporary details (e.g., "I am hungry now")
- Extract new people, objects, places mentioned
- Extract preferences and personal details
- Extract tasks, plans, or commitments
- Return empty array if no long-term facts found

Message:
{user_message}

Session Context:
{session_context}

Return only valid JSON:"""

        # Run synchronous Gemini call in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai_model.generate_content(extraction_prompt)
        )
        
        text = result.text.strip()
        
        # Parse JSON response
        import json
        # Try to extract JSON from response (may have markdown code blocks)
        if '```' in text:
            # Extract JSON from code block
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                text = text[start:end]
        
        data = json.loads(text)
        facts = data.get('facts', [])
        
        # Validate and format facts
        extracted_facts = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            
            fact_type = fact.get('type', 'general')
            value = fact.get('value', '').strip()
            tags = fact.get('tags', [])
            
            if value:
                extracted_facts.append({
                    'type': fact_type,
                    'content': value,
                    'tags': tags if isinstance(tags, list) else []
                })
        
        return extracted_facts
    except json.JSONDecodeError as error:
        print(f'⚠️  Failed to parse memory extraction JSON: {error}')
        return []
    except Exception as error:
        print(f'❌ Error extracting memory facts: {error}')
        return []


async def generate_batch_memory_summary(
    session_buffer: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate batch memory summary when session ends
    
    Args:
        session_buffer: Complete session buffer
    
    Returns:
        Dict with summary and extracted facts
    """
    if not genai_model and not initialize_gemini():
        print('⚠️  Gemini AI not configured for batch summary')
        return {'summary': '', 'facts': []}

    if not session_buffer or len(session_buffer) == 0:
        return {'summary': '', 'facts': []}

    try:
        # Build session transcript
        session_transcript = '\n'.join([
            f"User: {msg.get('userMessage', '')}\nAI: {msg.get('aiResponse', '')}"
            for msg in session_buffer
        ])

        # Batch summary prompt
        summary_prompt = f"""Summarize the entire conversation session into long-term facts.

Output JSON:
{{
  "summary": "brief session summary",
  "facts": [
    {{ "type": "person|place|preference|event|detail", "value": "fact content", "tags": [] }}
  ]
}}

Session:
{session_transcript}

Return only valid JSON:"""

        # Run synchronous Gemini call in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai_model.generate_content(summary_prompt)
        )
        
        text = result.text.strip()
        
        # Parse JSON response
        import json
        # Try to extract JSON from response
        if '```' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                text = text[start:end]
        
        data = json.loads(text)
        
        summary = data.get('summary', '')
        facts = data.get('facts', [])
        
        # Format facts
        formatted_facts = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            
            fact_type = fact.get('type', 'general')
            value = fact.get('value', '').strip()
            tags = fact.get('tags', [])
            
            if value:
                formatted_facts.append({
                    'type': fact_type,
                    'content': value,
                    'tags': tags if isinstance(tags, list) else []
                })
        
        return {
            'summary': summary,
            'facts': formatted_facts
        }
    except json.JSONDecodeError as error:
        print(f'⚠️  Failed to parse batch summary JSON: {error}')
        return {'summary': '', 'facts': []}
    except Exception as error:
        print(f'❌ Error generating batch summary: {error}')
        return {'summary': '', 'facts': []}


async def test_gemini_connection() -> Dict[str, Any]:
    """
    Test Gemini connection
    
    Returns:
        dict: Test result with success status
    """
    if not genai_model and not initialize_gemini():
        return {'success': False, 'error': 'Gemini not initialized'}

    try:
        # Run synchronous call in executor
        loop = asyncio.get_event_loop()
        test_result = await loop.run_in_executor(
            None,
            lambda: genai_model.generate_content('Say "Hello" in one word.')
        )
        
        response = test_result.text
        
        if not response or not response.strip():
            return {'success': False, 'error': 'Empty response from Gemini'}
        
        return {
            'success': True,
            'response': response,
            'model': 'gemini-2.5-flash-lite'
        }
    except Exception as error:
        return {
            'success': False,
            'error': str(error)
        }

