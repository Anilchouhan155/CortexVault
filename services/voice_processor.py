"""
Voice Processing Service
Handles: Speech-to-Text (Whisper), Translation (Hindi/English), Text-to-Speech, Fast Processing
"""

import os
import base64
import asyncio
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import httpx
from openai import OpenAI

# Load environment variables
load_dotenv()


class VoiceProcessor:
    """Voice processing service for STT, TTS, and translation"""
    
    def __init__(self):
        # OpenAI for Whisper (Speech-to-Text) and TTS
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        
        # Google APIs (fallback/optional)
        self.google_speech_api_key = os.getenv('GOOGLE_SPEECH_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.google_translate_api_key = os.getenv('GOOGLE_TRANSLATE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        # Temp directory
        self.temp_dir = Path(__file__).parent.parent / 'temp'
        self.temp_dir.mkdir(exist_ok=True)
        
        # Log which services are available
        if self.openai:
            print('✅ OpenAI Whisper available for speech-to-text')
        else:
            print('⚠️  OPENAI_API_KEY not set - Whisper not available')
        
        # Cleanup task will be started when event loop is running
    
    async def _cleanup_task(self) -> None:
        """Periodic cleanup of temp files"""
        while True:
            await asyncio.sleep(30 * 60)  # Every 30 minutes
            self.cleanup_temp_files()
    
    async def download_audio(self, media_id: str, media_url: str) -> str:
        """
        Download audio file from WhatsApp media URL
        
        Args:
            media_id: Media ID
            media_url: Media URL from WhatsApp
        
        Returns:
            str: Path to downloaded audio file
        """
        whatsapp_token = os.getenv('WHATSAPP_TOKEN')
        if not whatsapp_token:
            raise ValueError('WHATSAPP_TOKEN not configured')

        if not media_id or not media_url:
            raise ValueError('Media ID and URL are required')

        try:

            # Download from WhatsApp media URL
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    media_url,
                    headers={'Authorization': f'Bearer {whatsapp_token}'}
                )
                response.raise_for_status()
                audio_data = response.content

            # Save to temp file
            file_extension = self.get_file_extension(media_url) or '.ogg'
            temp_file_path = self.temp_dir / f'{media_id}{file_extension}'
            temp_file_path.write_bytes(audio_data)

            print(f'📥 Audio downloaded: {temp_file_path}')
            return str(temp_file_path)
        except Exception as error:
            print(f'❌ Error downloading audio: {error}')
            raise
    
    async def speech_to_text(self, audio_file_path: str, language_code: str = 'auto') -> Dict[str, Any]:
        """
        Convert audio to text using OpenAI Whisper
        
        Args:
            audio_file_path: Path to audio file
            language_code: Language code (auto-detect if 'auto')
        
        Returns:
            Dict[str, Any]: Transcription result with text, confidence, language
        """
        if not audio_file_path or not Path(audio_file_path).exists():
            raise ValueError(f'Audio file not found: {audio_file_path}')

        # Prefer OpenAI Whisper (better, faster, easier)
        if self.openai:
            try:
                return await self.speech_to_text_whisper(audio_file_path)
            except Exception as error:
                # If Whisper fails, try Google as fallback
                if 'Whisper' in str(error) and self.google_speech_api_key:
                    print('🔄 Whisper failed, trying Google Speech-to-Text...')
                    return await self.speech_to_text_google(audio_file_path, language_code)
                raise
        
        # Fallback to Google Speech-to-Text if OpenAI not available
        if self.google_speech_api_key:
            print('⚠️  Using Google Speech-to-Text (Whisper not available)')
            return await self.speech_to_text_google(audio_file_path, language_code)
        
        raise ValueError('No speech-to-text API configured. Set OPENAI_API_KEY or GOOGLE_SPEECH_API_KEY')
    
    async def speech_to_text_whisper(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Speech-to-text using OpenAI Whisper (primary method)
        
        Args:
            audio_file_path: Path to audio file
        
        Returns:
            Dict[str, Any]: Transcription result
        """
        if not self.openai_api_key:
            raise ValueError('OpenAI API not configured (set OPENAI_API_KEY)')

        if not Path(audio_file_path).exists():
            raise ValueError(f'Audio file not found: {audio_file_path}')

        try:

            print('🎤 Using OpenAI Whisper for speech-to-text...')

            # Read audio file
            audio_file = Path(audio_file_path)
            with open(audio_file, 'rb') as f:
                audio_data = f.read()

            # Run synchronous OpenAI call in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai.audio.transcriptions.create(
                    model='whisper-1',
                    file=(audio_file.name, audio_data, 'audio/ogg'),
                    response_format='verbose_json',
                    temperature=0.0
                )
            )

            # Extract transcription from response
            transcript = response.text or ''
            detected_lang = response.language or 'en'
            
            if not transcript:
                raise ValueError('No transcription text in response')
            
            # Map Whisper language codes to our format
            lang_map = {
                'hi': 'hi-IN',
                'en': 'en-US'
            }
            mapped_lang = lang_map.get(detected_lang, detected_lang)

            print(f'🎤 Whisper transcription: "{transcript}" ({mapped_lang})')
            
            return {
                'text': transcript,
                'confidence': 0.95,  # Whisper is very accurate
                'language': mapped_lang,
                'detectedLanguage': detected_lang
            }
        except Exception as error:
            print(f'❌ Whisper API error: {error}')
            if hasattr(error, 'response') and error.response:
                status = error.response.status_code if hasattr(error.response, 'status_code') else None
                if status == 401:
                    raise ValueError('OpenAI API key is invalid. Check your OPENAI_API_KEY.')
                elif status == 429:
                    raise ValueError('OpenAI API rate limit exceeded. Please try again later.')
            raise
    
    async def speech_to_text_google(self, audio_file_path: str, language_code: str = 'auto') -> Dict[str, Any]:
        """
        Speech-to-text using Google Speech-to-Text (fallback)
        
        Args:
            audio_file_path: Path to audio file
            language_code: Language code
        
        Returns:
            Dict[str, Any]: Transcription result
        """
        if not self.google_speech_api_key:
            raise ValueError('Google Speech API key not configured')

        if not Path(audio_file_path).exists():
            raise ValueError(f'Audio file not found: {audio_file_path}')

        try:

            # Read audio file
            audio_data = Path(audio_file_path).read_bytes()
            base64_audio = base64.b64encode(audio_data).decode('utf-8')

            # Use multi-language detection
            config = {
                'encoding': 'OGG_OPUS',
                'sampleRateHertz': 16000,
                'languageCode': 'hi-IN',
                'alternativeLanguageCodes': ['en-US'],
                'enableAutomaticPunctuation': True,
                'model': 'latest_short',
                'useEnhanced': True
            }

            async with httpx.AsyncClient(timeout=12.0) as client:
                response = await client.post(
                    f'https://speech.googleapis.com/v1/speech:recognize?key={self.google_speech_api_key}',
                    json={
                        'config': config,
                        'audio': {'content': base64_audio}
                    }
                )
                response.raise_for_status()
                result_data = response.json()

            if result_data.get('results') and len(result_data['results']) > 0:
                result = result_data['results'][0]
                transcript = result['alternatives'][0]['transcript']
                confidence = result['alternatives'][0].get('confidence', 0.8)
                detected_lang = result.get('languageCode', 'hi-IN')
                
                print(f'🎤 Google STT: "{transcript}" ({detected_lang}, confidence: {confidence:.2f})')
                return {
                    'text': transcript,
                    'confidence': confidence,
                    'language': detected_lang
                }

            raise ValueError('No transcription results')
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 403:
                raise ValueError('Google Speech-to-Text API not enabled. Use OPENAI_API_KEY for Whisper instead.')
            raise
        except Exception as error:
            print(f'❌ Google Speech-to-text API error: {error}')
            raise
    
    async def translate_text(self, text: str, source_language: str = 'auto') -> Dict[str, Any]:
        """
        Translate text to Hindi and English
        
        Args:
            text: Text to translate
            source_language: Source language code
        
        Returns:
            Dict[str, Any]: Translations in multiple languages
        """
        if not text or not text.strip():
            return {
                'original': text or '',
                'english': text or '',
                'hindi': text or '',
                'detectedLanguage': 'en'
            }

        if not self.google_translate_api_key:
            # If no translate API key, return original text
            print('⚠️ Translation API key not configured, skipping translation')
            return {
                'original': text,
                'english': text,
                'hindi': text,
                'detectedLanguage': 'en'
            }

        try:

            # Detect source language
            detected_lang = 'en'
            if source_language == 'auto':
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        detect_response = await client.post(
                            f'https://translation.googleapis.com/language/translate/v2/detect?key={self.google_translate_api_key}',
                            json={'q': text}
                        )
                        detect_response.raise_for_status()
                        detect_data = detect_response.json()
                        detected_lang = detect_data['data']['detections'][0][0]['language']
                except Exception:
                    # Try to infer from language code if provided
                    if isinstance(source_language, str) and '-' in source_language:
                        detected_lang = source_language.split('-')[0]
                    else:
                        detected_lang = 'en'
            else:
                # Extract language code (e.g., 'hi-IN' -> 'hi')
                detected_lang = source_language.split('-')[0] if '-' in source_language else source_language

            translations = {'original': text, 'detectedLanguage': detected_lang}

            # Translate to English if not already
            if detected_lang != 'en':
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        en_response = await client.post(
                            f'https://translation.googleapis.com/language/translate/v2?key={self.google_translate_api_key}',
                            json={
                                'q': text,
                                'source': detected_lang,
                                'target': 'en',
                                'format': 'text'
                            }
                        )
                        en_response.raise_for_status()
                        en_data = en_response.json()
                        translations['english'] = en_data['data']['translations'][0]['translatedText']
                except Exception:
                    translations['english'] = text  # Fallback to original
            else:
                translations['english'] = text

            # Translate to Hindi
            if detected_lang != 'hi':
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        hi_response = await client.post(
                            f'https://translation.googleapis.com/language/translate/v2?key={self.google_translate_api_key}',
                            json={
                                'q': text,
                                'source': detected_lang,
                                'target': 'hi',
                                'format': 'text'
                            }
                        )
                        hi_response.raise_for_status()
                        hi_data = hi_response.json()
                        translations['hindi'] = hi_data['data']['translations'][0]['translatedText']
                except Exception:
                    translations['hindi'] = text  # Fallback to original
            else:
                translations['hindi'] = text

            print(f'🌐 Translation: EN="{translations["english"]}" | HI="{translations["hindi"]}"')
            return translations
        except Exception as error:
            print(f'❌ Translation error: {error}')
            # Return original text if translation fails
            return {
                'original': text,
                'english': text,
                'hindi': text,
                'detectedLanguage': 'en'
            }
    
    async def process_voice_message(self, media_id: str, media_url: str) -> Dict[str, Any]:
        """
        Process voice message: Download -> STT -> Translate -> Return
        
        Args:
            media_id: Media ID from WhatsApp
            media_url: Media URL from WhatsApp
        
        Returns:
            Dict[str, Any]: Processing result with transcription and translations
        """
        if not media_id or not media_url:
            return {
                'success': False,
                'error': 'Media ID and URL are required',
                'processingTime': 0
            }

        start_time = datetime.now()
        audio_file_path = None

        try:
            print(f'🎙️ Processing voice message: {media_id}')

            # Step 1: Download audio (required first)
            audio_file_path = await self.download_audio(media_id, media_url)
            download_time = (datetime.now() - start_time).total_seconds() * 1000

            # Step 2: Speech-to-text
            stt_result = await self.speech_to_text(audio_file_path)
            stt_time = (datetime.now() - start_time).total_seconds() * 1000 - download_time

            # Step 3: Translate
            translations = await self.translate_text(stt_result['text'], stt_result.get('language', 'auto'))
            total_time = (datetime.now() - start_time).total_seconds() * 1000

            print(f'⏱️ Voice processing complete in {total_time:.0f}ms (download: {download_time:.0f}ms, STT: {stt_time:.0f}ms)')

            return {
                'success': True,
                'originalText': stt_result['text'],
                'translations': translations,
                'confidence': stt_result.get('confidence', 0.8),
                'processingTime': total_time,
                'detectedLanguage': translations.get('detectedLanguage') or stt_result.get('language', 'en').split('-')[0] if isinstance(stt_result.get('language'), str) else 'en',
                'audioFilePath': audio_file_path  # Keep for voice analysis
            }
        except Exception as error:
            print(f'❌ Voice processing failed: {error}')
            return {
                'success': False,
                'error': str(error),
                'processingTime': (datetime.now() - start_time).total_seconds() * 1000
            }
    
    def get_file_extension(self, url: str) -> Optional[str]:
        """
        Get file extension from URL
        
        Args:
            url: URL string
        
        Returns:
            Optional[str]: File extension or None
        """
        if not url:
            return None

        try:
            parsed = urlparse(url)
            path = parsed.path
            if '.' in path:
                return '.' + path.split('.')[-1]
            return None
        except Exception:
            return None
    
    async def text_to_speech(
        self,
        text: str,
        language_code: str = 'en-US',
        voice_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert text to speech using OpenAI TTS or Google TTS
        
        Args:
            text: Text to convert
            language_code: Language code
            voice_params: Voice parameters (speakingRate, pitch, volume)
        
        Returns:
            Dict[str, Any]: TTS result with audio file path
        """
        if not text or not text.strip():
            raise ValueError('Text is required for TTS')

        if voice_params is None:
            voice_params = {}
        
        # Prefer OpenAI TTS (better quality, easier setup)
        if self.openai:
            try:
                return await self.text_to_speech_openai(text, language_code, voice_params)
            except Exception as error:
                # If OpenAI TTS fails, try Google as fallback
                if 'OpenAI' in str(error) and self.google_speech_api_key:
                    print('🔄 OpenAI TTS failed, trying Google TTS...')
                    return await self.text_to_speech_google(text, language_code, voice_params)
                raise
        
        # Fallback to Google TTS
        if self.google_speech_api_key:
            print('⚠️ Using Google TTS (OpenAI TTS not available)')
            return await self.text_to_speech_google(text, language_code, voice_params)
        
        raise ValueError('No text-to-speech API configured. Set OPENAI_API_KEY or GOOGLE_SPEECH_API_KEY')
    
    async def text_to_speech_openai(
        self,
        text: str,
        language_code: str = 'en-US',
        voice_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Text-to-speech using OpenAI TTS (primary method)
        
        Args:
            text: Text to convert
            language_code: Language code
            voice_params: Voice parameters
        
        Returns:
            Dict[str, Any]: TTS result
        """
        if voice_params is None:
            voice_params = {}
        
        if not self.openai:
            raise ValueError('OpenAI API not configured')

        if not text or not text.strip():
            raise ValueError('Text is required for TTS')

        try:

            # Map language codes for OpenAI TTS
            lang_map = {
                'hi': 'hi',
                'hi-IN': 'hi',
                'en': 'en',
                'en-US': 'en',
                'en-GB': 'en'
            }
            tts_lang = lang_map.get(language_code, 'en')

            # Select voice based on language
            voice = 'alloy' if tts_lang == 'hi' else 'shimmer'

            print(f'🔊 Using OpenAI TTS ({voice} voice, {tts_lang})...')

            # Run synchronous OpenAI call in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai.audio.speech.create(
                    model='tts-1',
                    voice=voice,
                    input=text,
                    speed=voice_params.get('speakingRate', 1.0)
                )
            )

            # Get audio buffer
            audio_buffer = response.content
            
            # Save to temp file
            temp_file_path = self.temp_dir / f'tts_openai_{int(datetime.now().timestamp() * 1000)}.mp3'
            temp_file_path.write_bytes(audio_buffer)

            print(f'🔊 OpenAI TTS generated: {text[:50]}... ({tts_lang})')
            return {
                'success': True,
                'audioFilePath': str(temp_file_path),
                'language': language_code,
                'voiceName': voice,
                'format': 'mp3'
            }
        except Exception as error:
            print(f'❌ OpenAI TTS error: {error}')
            raise
    
    async def text_to_speech_google(
        self,
        text: str,
        language_code: str = 'en-US',
        voice_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Text-to-speech using Google TTS (fallback method)
        
        Args:
            text: Text to convert
            language_code: Language code
            voice_params: Voice parameters
        
        Returns:
            Dict[str, Any]: TTS result
        """
        if voice_params is None:
            voice_params = {}
        
        if not self.google_speech_api_key:
            raise ValueError('Google Speech API key not configured')

        if not text or not text.strip():
            raise ValueError('Text is required for TTS')

        try:

            # Map language codes for TTS
            language_map = {
                'hi': 'hi-IN',
                'hi-IN': 'hi-IN',
                'en': 'en-US',
                'en-US': 'en-US',
                'en-GB': 'en-GB'
            }

            tts_language = language_map.get(language_code, language_code) or 'en-US'
            
            # Select voice based on language
            voice_name = 'hi-IN-Standard-A' if tts_language.startswith('hi') else 'en-US-Neural2-D'

            # Use Google Text-to-Speech REST API
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f'https://texttospeech.googleapis.com/v1/text:synthesize?key={self.google_speech_api_key}',
                    json={
                        'input': {'text': text},
                        'voice': {
                            'languageCode': tts_language,
                            'name': voice_name,
                            'ssmlGender': 'FEMALE' if tts_language.startswith('hi') else 'NEUTRAL'
                        },
                        'audioConfig': {
                            'audioEncoding': 'OGG_OPUS',
                            'speakingRate': voice_params.get('speakingRate', 1.0),
                            'pitch': voice_params.get('pitch', 0.0),
                            'volumeGainDb': voice_params.get('volume', 0.0)
                        }
                    }
                )
                response.raise_for_status()
                result_data = response.json()

            if result_data.get('audioContent'):
                # Decode base64 audio
                audio_buffer = base64.b64decode(result_data['audioContent'])
                
                # Save to temp file
                temp_file_path = self.temp_dir / f'tts_{int(datetime.now().timestamp() * 1000)}.ogg'
                temp_file_path.write_bytes(audio_buffer)

                print(f'🔊 Text-to-speech generated: {text[:50]}... ({tts_language})')
                return {
                    'success': True,
                    'audioFilePath': str(temp_file_path),
                    'language': tts_language,
                    'voiceName': voice_name
                }

            raise ValueError('No audio content in response')
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 403:
                raise ValueError('Text-to-Speech API not enabled or API key lacks permissions. Enable Cloud Text-to-Speech API in Google Cloud Console.')
            raise
        except Exception as error:
            print(f'❌ Text-to-speech error: {error}')
            raise
    
    async def analyze_voice_characteristics(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Analyze voice characteristics from audio (pitch, speed, tone)
        
        Args:
            audio_file_path: Path to audio file
        
        Returns:
            Dict[str, Any]: Voice parameters to match
        """
        # This is a simplified version - in production, you'd use audio analysis libraries
        # For now, return default parameters that work well
        if not audio_file_path or not Path(audio_file_path).exists():
            print('⚠️ Audio file not found for voice analysis, using defaults')
        
        return {
            'speakingRate': 1.0,  # Normal speed
            'pitch': 0.0,  # Neutral pitch
            'volume': 0.0  # Normal volume
        }
    
    def cleanup_temp_files(self) -> None:
        """Clean up old temp files (run periodically)"""
        try:
            if not self.temp_dir.exists():
                return
            
            now = datetime.now().timestamp()
            max_age = 60 * 60  # 1 hour in seconds

            for file_path in self.temp_dir.iterdir():
                if file_path.is_file():
                    file_stat = file_path.stat()
                    file_age = now - file_stat.st_mtime
                    if file_age > max_age:
                        file_path.unlink()
                        print(f'🧹 Cleaned up temp file: {file_path.name}')
        except Exception as error:
            print(f'⚠️ Cleanup error: {error}')


# Create singleton instance
voice_processor = VoiceProcessor()

# Start cleanup task when module is imported (if event loop exists)
try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(voice_processor._cleanup_task())
except RuntimeError:
    # No event loop yet, will be started when app starts
    pass

