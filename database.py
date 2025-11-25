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

