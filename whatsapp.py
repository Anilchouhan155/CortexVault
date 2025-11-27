"""
WhatsApp Webhook Handler
Handles incoming messages, sends responses, and manages WhatsApp API interactions
"""

import os
import re
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import httpx
from ai import generate_ai_response, generate_batch_memory_summary
from database import save_conversation, get_recent_conversations, get_or_create_user
from services.memory_service import (
    process_session_message,
    extract_memory_facts,
    save_extracted_memories
)

# Load environment variables
load_dotenv()

# WhatsApp API configuration
WHATSAPP_TOKEN = os.getenv('WHATSAPP_TOKEN')
WHATSAPP_URL = os.getenv('WHATSAPP_URL', 'https://graph.facebook.com/v18.0')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
WHATSAPP_VERIFY_TOKEN = os.getenv('WHATSAPP_VERIFY_TOKEN')

# Track processed message IDs to prevent duplicates
processed_messages = {}
MESSAGE_ID_TTL = 24 * 60 * 60  # 24 hours in seconds

# Clean up old message IDs periodically
async def cleanup_processed_messages() -> None:
    """Periodically clean up old message IDs"""
    while True:
        await asyncio.sleep(3600)  # Every hour
        current_time = datetime.now().timestamp()
        expired_ids = [
            msg_id for msg_id, timestamp in processed_messages.items()
            if current_time - timestamp > MESSAGE_ID_TTL
        ]
        for msg_id in expired_ids:
            del processed_messages[msg_id]
        if len(processed_messages) > 1000:
            processed_messages.clear()

# Start cleanup task (will be started when event loop is running)
_cleanup_task_started = False

def start_cleanup_task() -> None:
    """Start cleanup task (called from app startup)"""
    global _cleanup_task_started
    if not _cleanup_task_started:
        asyncio.create_task(cleanup_processed_messages())
        _cleanup_task_started = True


async def verify_webhook(mode: str, token: str, challenge: str) -> Optional[str]:
    """
    Verify webhook (GET request from WhatsApp)
    
    Args:
        mode: hub.mode from query params
        token: hub.verify_token from query params
        challenge: hub.challenge from query params
    
    Returns:
        Optional[str]: Challenge string if verified, None otherwise
    """
    print('🔍 Webhook verification request:', {
        'mode': mode,
        'tokenProvided': bool(token),
        'challengeProvided': bool(challenge)
    })

    if not WHATSAPP_VERIFY_TOKEN:
        print('❌ WHATSAPP_VERIFY_TOKEN not configured')
        return None

    if mode == 'subscribe' and token == WHATSAPP_VERIFY_TOKEN:
        print('✅ Webhook verified successfully')
        return challenge
    
    print('❌ Webhook verification failed - token mismatch')
    return None


async def handle_message(body: Dict[str, Any]) -> None:
    """
    Handle incoming messages (POST request from WhatsApp)
    
    Args:
        body: Webhook payload from WhatsApp
    """
    if not body:
        print('⚠️ Empty webhook body received')
        return

    print('📱 Received webhook:', {
        'object': body.get('object'),
        'entryCount': len(body.get('entry', []))
    })

    if body.get('object') != 'whatsapp_business_account':
        print(f'⚠️ Received non-WhatsApp webhook: {body.get("object")}')
        return

    entries = body.get('entry', [])
    if not entries:
        print('⚠️ No entries in webhook payload')
        return

    try:
        for entry in entries:
            changes = entry.get('changes', [])
            for change in changes:
                if change.get('field') == 'messages' and change.get('value'):
                    value = change.get('value')
                    messages = value.get('messages', [])
                    contacts = value.get('contacts', [])

                    for message in messages:
                        # Check for duplicate messages
                        message_id = message.get('id')
                        if message_id and message_id in processed_messages:
                            print(f'⏭️ Duplicate message ignored: {message_id}')
                            continue

                        # Mark message as processed
                        if message_id:
                            processed_messages[message_id] = datetime.now().timestamp()

                        # Find contact for this message
                        contact = next(
                            (c for c in contacts if c.get('wa_id') == message.get('from')),
                            contacts[0] if contacts else None
                        )
                        
                        # Process message asynchronously (don't await to avoid timeout)
                        asyncio.create_task(process_message(message, contact))
    except Exception as error:
        print(f'❌ Error handling webhook: {error}')


async def process_message(message: Dict[str, Any], contact: Optional[Dict[str, Any]]) -> None:
    """
    Process individual message
    
    Args:
        message: Message object from WhatsApp
        contact: Contact object from WhatsApp
    """
    # Validate message structure
    if not message or not message.get('from'):
        print('⚠️ Invalid message structure:', message)
        return

    try:

        user_phone = message.get('from')
        user_name = contact.get('profile', {}).get('name', 'User') if contact else 'User'
        
        user_message = None
        message_type = 'text'
        voice_processing_result = None

        # Import voice processor here to avoid circular imports
        from services.voice_processor import VoiceProcessor
        voice_processor = VoiceProcessor()

        # Handle different message types
        if message.get('type') == 'text':
            text_data = message.get('text', {})
            if not text_data or not text_data.get('body'):
                print('⚠️ Text message missing body')
                return
            user_message = text_data.get('body', '').strip()
            message_type = 'text'
        elif message.get('type') in ['audio', 'voice']:
            # Handle voice messages
            print(f'🎙️ Voice message received from {user_name}')
            
            try:
                media_id = message.get('audio', {}).get('id') or message.get('voice', {}).get('id')
                media_url = await get_media_url(media_id)
                
                if not media_url:
                    raise Exception('Could not get media URL')

                # Process voice: Download -> STT -> Translate
                voice_processing_result = await voice_processor.process_voice_message(media_id, media_url)
                
                if voice_processing_result.get('success'):
                    # Use original text in detected language for AI processing
                    detected_lang = voice_processing_result.get('translations', {}).get('detectedLanguage', 'en')
                    user_message = voice_processing_result.get('originalText')
                    message_type = 'voice'
                    print(f'✅ Voice processed: "{user_message}" ({detected_lang})')
                    
                    # Store detected language for later use
                    voice_processing_result['detectedLanguage'] = detected_lang
                else:
                    raise Exception(voice_processing_result.get('error', 'Voice processing failed'))
            except Exception as error:
                print(f'❌ Error processing voice message: {error}')
                
                # Provide helpful error message
                error_message = "Hey! I couldn't understand your voice message. Can you try sending it as text? 😅"
                
                error_str = str(error)
                if '403' in error_str:
                    error_message = "Hey! Voice processing isn't set up yet. Can you send that as text for now? 😅"
                    print('💡 Fix: Enable Cloud Speech-to-Text API in Google Cloud Console')
                elif 'API not enabled' in error_str:
                    error_message = "Hey! Voice features need setup. Send as text for now? 😅"
                    print('💡 Fix: Enable Speech-to-Text and Text-to-Speech APIs')
                
                await send_whatsapp_message(user_phone, error_message)
                return
        else:
            print(f'ℹ️ Skipping unsupported message type: {message.get("type")}')
            return

        # Validate message is not empty
        if not user_message or len(user_message) == 0:
            print('ℹ️ Empty message received, ignoring')
            return

        # Validate message length (max 4096 chars for WhatsApp)
        if len(user_message) > 4096:
            print('⚠️ Message too long, truncating')
            user_message = user_message[:4096]

        print(f'📱 Message from {user_name} ({user_phone}): {user_message[:100]}{"..." if len(user_message) > 100 else ""}')

        # Get or create user ID for memory system
        user_id = await get_or_create_user(user_phone, user_name)
        if not user_id:
            # Fallback to phone number if user creation fails
            user_id = user_phone

        # Process session and retrieve memories (with fallback)
        memory_context = None
        try:
            memory_context = await process_session_message(
                user_id=user_id,
                user_message=user_message,
                ai_response=''  # Will be filled after generation
            )
        except Exception as error:
            print(f'⚠️  Error processing session/memory (falling back to basic mode): {error}')
            memory_context = None

        # Get short-term memory (session buffer)
        short_term_memory = memory_context.get('short_term_memory', []) if memory_context else []
        
        # Get long-term memories if retrieved
        long_term_memories = memory_context.get('long_term_memories', []) if memory_context else []
        
        # Log memory retrieval for debugging
        if long_term_memories:
            print(f'📝 Retrieved {len(long_term_memories)} memories for query: "{user_message[:50]}"')
            for i, mem in enumerate(long_term_memories[:3], 1):
                print(f'   {i}. {mem.get("content", "")[:60]} (similarity: {mem.get("similarity", 0):.3f})')
        else:
            print(f'⚠️  No memories retrieved for query: "{user_message[:50]}"')
        
        # Get recent conversation context (fallback if memory system fails)
        recent_chats = await get_recent_conversations(user_phone, 5)
        if not short_term_memory and recent_chats:
            short_term_memory = recent_chats
        
        # Generate AI response with timeout and memory context
        ai_response = None
        try:
            start_time = datetime.now()
            
            # Determine target language for AI response
            target_language = 'auto'
            if message_type == 'voice' and voice_processing_result:
                detected_lang = voice_processing_result.get('detectedLanguage', 'en')
                if detected_lang == 'hi' or detected_lang.startswith('hi'):
                    target_language = 'hi'
                else:
                    target_language = 'en'
                print(f'🌍 Responding in detected language: {target_language}')
            
            # Race between AI response and timeout
            response_task = asyncio.create_task(
                generate_ai_response(
                    user_message=user_message,
                    conversation_history=short_term_memory,
                    user_name=user_name,
                    target_language=target_language,
                    retrieved_memories=long_term_memories
                )
            )
            timeout_task = asyncio.create_task(asyncio.sleep(20))  # 20 second timeout
            
            done, pending = await asyncio.wait(
                [response_task, timeout_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Check which task completed
            if timeout_task in done:
                raise Exception('AI response timeout')
            
            # Get result from response task
            ai_response = await response_task
            duration = (datetime.now() - start_time).total_seconds() * 1000
            print(f'⏱️ AI response generated in {duration:.0f}ms')
        except Exception as error:
            print(f'❌ Error generating AI response: {error}')
            if 'timeout' in str(error).lower():
                print('⚠️ Gemini API took longer than 20 seconds to respond')
            ai_response = "Hey! Something went wrong on my end. Can you try again? 😅"
        
        # Update session with AI response
        if memory_context:
            try:
                from services.memory_service import update_session
                await update_session(user_id, {
                    'userMessage': user_message,
                    'aiResponse': ai_response,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as error:
                print(f'⚠️  Error updating session: {error}')
        
        # Extract and save memories (hot path)
        try:
            facts = await extract_memory_facts(
                user_message=user_message,
                session_buffer=short_term_memory,
                ai_response=ai_response
            )
            
            if facts and len(facts) > 0:
                saved_count = await save_extracted_memories(user_id, facts)
                if saved_count > 0:
                    print(f'💾 Saved {saved_count} memory fact(s)')
        except Exception as error:
            print(f'⚠️  Error extracting/saving memories: {error}')
        
        # Save conversation
        saved = await save_conversation({
            'phone': user_phone,
            'userName': user_name,
            'userMessage': user_message,
            'aiResponse': ai_response,
            'timestamp': datetime.now().isoformat(),
            'messageType': message_type,
            'voiceMetadata': {
                'originalText': voice_processing_result.get('originalText'),
                'translations': voice_processing_result.get('translations'),
                'confidence': voice_processing_result.get('confidence')
            } if message_type == 'voice' and voice_processing_result else None
        })

        if not saved:
            print('❌ Failed to save conversation')

        # Send response back to WhatsApp
        try:
            # If original message was voice, send voice response
            if message_type == 'voice' and voice_processing_result:
                detected_lang = voice_processing_result.get('detectedLanguage', 'en')
                lang_code = 'hi-IN' if detected_lang == 'hi' else 'en-US'
                
                # Analyze voice characteristics
                voice_params = {
                    'speakingRate': 1.0,
                    'pitch': 0.0,
                    'volume': 0.0
                }
                
                if voice_processing_result.get('audioFilePath'):
                    try:
                        voice_params = await voice_processor.analyze_voice_characteristics(
                            voice_processing_result.get('audioFilePath')
                        )
                    except Exception:
                        print('⚠️ Voice analysis failed, using defaults')
                
                # Convert AI response to speech
                tts_result = await voice_processor.text_to_speech(ai_response, lang_code, voice_params)
                
                if tts_result.get('success'):
                    # Send voice message
                    await send_whatsapp_voice_message(
                        user_phone,
                        tts_result.get('audioFilePath'),
                        tts_result.get('format', 'ogg')
                    )
                    print(f'✅ Voice response sent to {user_name} in {detected_lang}')
                    
                    # Clean up TTS file and original audio file after sending
                    async def cleanup_files():
                        await asyncio.sleep(5)
                        try:
                            tts_path = tts_result.get('audioFilePath')
                            if tts_path and Path(tts_path).exists():
                                Path(tts_path).unlink()
                            orig_path = voice_processing_result.get('audioFilePath')
                            if orig_path and Path(orig_path).exists():
                                Path(orig_path).unlink()
                        except Exception:
                            pass  # Ignore cleanup errors
                    
                    asyncio.create_task(cleanup_files())
                else:
                    # Fallback to text if TTS fails
                    await send_whatsapp_message(user_phone, ai_response)
                    print(f'✅ Text response sent (TTS failed) to {user_name}')
            else:
                # Send text response for text messages
                await send_whatsapp_message(user_phone, ai_response)
                print(f'✅ Response sent to {user_name}')
        except Exception as error:
            print(f'❌ Failed to send WhatsApp message: {error}')
            # Try fallback to text
            try:
                await send_whatsapp_message(user_phone, ai_response)
                print(f'✅ Fallback text response sent to {user_name}')
            except Exception as fallback_error:
                print(f'❌ Fallback also failed: {fallback_error}')
    except Exception as error:
        print(f'❌ Error processing message: {error}')
        # Try to send error message to user
        try:
            if message and message.get('from'):
                await send_whatsapp_message(
                    message.get('from'),
                    "Hey! Something went wrong. Can you try sending that again?"
                )
        except Exception as send_error:
            print(f'❌ Failed to send error message: {send_error}')


async def send_whatsapp_message(to: str, message: str) -> Dict[str, Any]:
    """
    Send message via WhatsApp API
    
    Args:
        to: Phone number (with country code, no +)
        message: Message text
    
    Returns:
        Dict[str, Any]: API response
    """
    # Validate inputs
    if not to or not message:
        raise ValueError('Missing required parameters: to and message')

    if not WHATSAPP_TOKEN:
        raise ValueError('WHATSAPP_TOKEN not configured')

    if not WHATSAPP_PHONE_NUMBER_ID:
        raise ValueError('WHATSAPP_PHONE_NUMBER_ID not configured')

    # Validate phone number format
    if not re.match(r'^\d{10,15}$', to):
        raise ValueError(f'Invalid phone number format: {to}')

    # Truncate message if too long (WhatsApp limit is 4096 chars)
    truncated_message = message[:4090] + '...' if len(message) > 4096 else message

    try:

        url = f'{WHATSAPP_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages'
        
        data = {
            'messaging_product': 'whatsapp',
            'recipient_type': 'individual',
            'to': to,
            'type': 'text',
            'text': {
                'preview_url': False,
                'body': truncated_message
            }
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                json=data,
                headers={
                    'Authorization': f'Bearer {WHATSAPP_TOKEN}',
                    'Content-Type': 'application/json'
                }
            )
            response.raise_for_status()
            result = response.json()
            print(f'✅ Message sent successfully: {result.get("messages", [{}])[0].get("id", "unknown")}')
            return result
    except httpx.HTTPStatusError as error:
        print(f'❌ WhatsApp API error: {error.response.status_code} - {error.response.text}')
        raise
    except Exception as error:
        print(f'❌ Error sending WhatsApp message: {error}')
        raise


async def get_media_url(media_id: str) -> Optional[str]:
    """
    Get media URL from WhatsApp API
    
    Args:
        media_id: Media ID from WhatsApp
    
    Returns:
        Optional[str]: Media URL or None
    """
    if not media_id or not WHATSAPP_TOKEN:
        return None

    try:

        url = f'{WHATSAPP_URL}/{media_id}'
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                url,
                headers={'Authorization': f'Bearer {WHATSAPP_TOKEN}'}
            )
            response.raise_for_status()
            return response.json().get('url')
    except Exception as error:
        print(f'❌ Error getting media URL: {error}')
        return None


async def upload_media(audio_file_path: str, audio_format: str = 'ogg') -> str:
    """
    Upload media to WhatsApp and get media ID
    
    Args:
        audio_file_path: Path to audio file
        audio_format: Format ('ogg' or 'mp3')
    
    Returns:
        str: Media ID
    """
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        raise ValueError('WhatsApp credentials not configured')

    if not audio_file_path or not Path(audio_file_path).exists():
        raise ValueError(f'Audio file not found: {audio_file_path}')

    try:

        # Read audio file
        audio_data = Path(audio_file_path).read_bytes()
        
        # Determine content type
        content_type = 'audio/mpeg' if audio_format == 'mp3' else 'audio/ogg; codecs=opus'
        filename = 'voice.mp3' if audio_format == 'mp3' else 'voice.ogg'
        
        url = f'{WHATSAPP_URL}/{WHATSAPP_PHONE_NUMBER_ID}/media'
        
        files = {
            'file': (filename, audio_data, content_type)
        }
        data = {
            'messaging_product': 'whatsapp',
            'type': 'audio'
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                url,
                files=files,
                data=data,
                headers={'Authorization': f'Bearer {WHATSAPP_TOKEN}'}
            )
            response.raise_for_status()
            return response.json().get('id')
    except Exception as error:
        print(f'❌ Error uploading media: {error}')
        raise


async def send_whatsapp_voice_message(to: str, audio_file_path: str, audio_format: str = 'ogg') -> Dict[str, Any]:
    """
    Send voice message via WhatsApp API
    
    Args:
        to: Phone number
        audio_file_path: Path to audio file
        audio_format: Format ('ogg' or 'mp3')
    
    Returns:
        Dict[str, Any]: API response
    """
    if not to or not audio_file_path:
        raise ValueError('Missing required parameters: to and audio_file_path')

    if not WHATSAPP_TOKEN:
        raise ValueError('WHATSAPP_TOKEN not configured')

    if not WHATSAPP_PHONE_NUMBER_ID:
        raise ValueError('WHATSAPP_PHONE_NUMBER_ID not configured')

    # Validate phone number format
    if not re.match(r'^\d{10,15}$', to):
        raise ValueError(f'Invalid phone number format: {to}')

    if not Path(audio_file_path).exists():
        raise ValueError(f'Audio file not found: {audio_file_path}')

    try:

        # Upload media first
        media_id = await upload_media(audio_file_path, audio_format)
        
        # Send voice message
        url = f'{WHATSAPP_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages'
        data = {
            'messaging_product': 'whatsapp',
            'recipient_type': 'individual',
            'to': to,
            'type': 'audio',
            'audio': {
                'id': media_id
            }
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                json=data,
                headers={
                    'Authorization': f'Bearer {WHATSAPP_TOKEN}',
                    'Content-Type': 'application/json'
                }
            )
            response.raise_for_status()
            result = response.json()
            print(f'✅ Voice message sent successfully: {result.get("messages", [{}])[0].get("id", "unknown")}')
            return result
    except httpx.HTTPStatusError as error:
        print(f'❌ WhatsApp API error: {error.response.status_code} - {error.response.text}')
        raise
    except Exception as error:
        print(f'❌ Error sending voice message: {error}')
        raise


def get_webhook_status() -> Dict[str, Any]:
    """
    Get webhook status
    
    Returns:
        Dict[str, Any]: Webhook configuration status
    """
    return {
        'configured': bool(WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_VERIFY_TOKEN),
        'hasToken': bool(WHATSAPP_TOKEN),
        'hasPhoneNumberId': bool(WHATSAPP_PHONE_NUMBER_ID),
        'hasVerifyToken': bool(WHATSAPP_VERIFY_TOKEN),
        'processedMessages': len(processed_messages),
        'url': WHATSAPP_URL,
        'voiceProcessingEnabled': bool(os.getenv('GOOGLE_SPEECH_API_KEY') or os.getenv('GEMINI_API_KEY'))
    }

