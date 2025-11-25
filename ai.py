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
    target_language: str = 'en'
) -> str:
    """
    Generate AI response with personality
    
    Args:
        user_message: User's message text
        conversation_history: List of previous conversations
        user_name: Name of the user
        target_language: Language code ('en', 'hi', 'auto')
    
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
        # Build context from recent conversations
        context = ''
        if conversation_history:
            context = '\n\n'.join([
                f"{user_name}: {conv.get('userMessage', conv.get('user_message', ''))}\n"
                f"You: {conv.get('aiResponse', conv.get('ai_response', ''))}"
                for conv in conversation_history
            ])

        # Language-specific instructions
        language_instructions = {
            'hi': 'Respond in Hindi (हिंदी). Use natural, conversational Hindi. Keep it casual and friendly.',
            'en': 'Respond in English. Keep it casual and friendly.',
            'auto': 'Respond in the same language as the user\'s message. Match their language naturally.'
        }

        lang_instruction = language_instructions.get(target_language, language_instructions['auto'])

        # Friend-first personality prompt
        # Build context prefix (extract to avoid backslash in f-string)
        context_prefix = ''
        if context:
            context_prefix = f'Previous conversation:\n{context}\n\n'
        
        system_prompt = f"""You are a helpful AI friend. You:
- Talk casually like a close friend
- Never use "please" or formal language  
- Remember previous conversations
- Show genuine interest in the user's life
- Are supportive and caring
- Keep responses under 2 sentences for WhatsApp
- Use the user's name ({user_name}) occasionally but naturally
- Be conversational and warm, not robotic
- {lang_instruction}

{context_prefix}User's message: {user_message}

Respond as their AI friend in the same language:"""

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

