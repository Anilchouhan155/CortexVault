#!/usr/bin/env python3
"""
NexaBrain MVP Setup Verification Script
Checks if everything is ready for the backend to run
"""

import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Colors for terminal output
class Colors:
    RESET = '\033[0m'
    GREEN = '\033[32m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'

def log(message: str, color: str = Colors.RESET) -> None:
    """Print colored message"""
    print(f"{color}{message}{Colors.RESET}")

def log_test(name: str, status: str, message: str = '') -> bool:
    """Log test result"""
    icon = '✅' if status == 'PASS' else '❌' if status == 'FAIL' else '⚠️'
    color = Colors.GREEN if status == 'PASS' else Colors.RED if status == 'FAIL' else Colors.YELLOW
    log(f"{icon} {name}{': ' + message if message else ''}", color)
    return status == 'PASS'

def log_section(title: str) -> None:
    """Log section header"""
    print()
    log('=' * 60, Colors.CYAN)
    log(f"  {title}", Colors.CYAN)
    log('=' * 60, Colors.CYAN)
    print()

# Track results
all_good = True
has_required = True

# 1. Check Python version
log_section('1. Python Version Check')
python_version = sys.version_info
if python_version >= (3, 8):
    log_test('Python version', 'PASS', f'{python_version.major}.{python_version.minor}.{python_version.micro}')
else:
    log_test('Python version', 'FAIL', f'{python_version.major}.{python_version.minor} (requires 3.8+)')
    all_good = False

# 2. Check dependencies
log_section('2. Checking Dependencies')
required_packages: List[str] = [
    'fastapi',
    'uvicorn',
    'python-dotenv',
    'httpx',
    'supabase',
    'google.generativeai',
    'openai',
    'pydantic',
    'aiofiles'
]

missing_packages: List[str] = []
for package in required_packages:
    try:
        __import__(package.replace('.', '_') if '.' in package else package)
        log_test(f'Package: {package}', 'PASS')
    except ImportError:
        log_test(f'Package: {package}', 'FAIL', 'NOT INSTALLED')
        missing_packages.append(package)
        all_good = False

if missing_packages:
    print(f"\n   Run: pip install {' '.join(missing_packages)}")

# 3. Check required files
log_section('3. Checking Required Files')
base_dir = Path(__file__).parent.parent
required_files = [
    'app.py',
    'whatsapp.py',
    'ai.py',
    'database.py',
    'services/voice_processor.py'
]

for file_path in required_files:
    full_path = base_dir / file_path
    if full_path.exists():
        log_test(f'File: {file_path}', 'PASS')
    else:
        log_test(f'File: {file_path}', 'FAIL', 'MISSING')
        all_good = False

# 4. Check directories
log_section('4. Checking Directories')
temp_dir = base_dir / 'temp'
if temp_dir.exists():
    log_test('temp/ directory', 'PASS', 'exists')
else:
    log_test('temp/ directory', 'PASS', 'will be created automatically')

# 5. Check environment variables
log_section('5. Checking Environment Variables')
required_env_vars = {
    'WHATSAPP_TOKEN': 'Required for WhatsApp API',
    'WHATSAPP_PHONE_NUMBER_ID': 'Required for WhatsApp API',
    'WHATSAPP_VERIFY_TOKEN': 'Required for webhook verification',
    'GEMINI_API_KEY': 'Required for AI responses'
}

optional_env_vars = {
    'OPENAI_API_KEY': 'Optional (for Whisper STT and TTS)',
    'GOOGLE_SPEECH_API_KEY': 'Optional (fallback for STT/TTS)',
    'GOOGLE_TRANSLATE_API_KEY': 'Optional (for translation)',
    'SUPABASE_URL': 'Optional (for database storage)',
    'SUPABASE_SERVICE_ROLE_KEY': 'Optional (for database storage)'
}

for var_name, description in required_env_vars.items():
    if os.getenv(var_name):
        log_test(f'Env var: {var_name}', 'PASS')
    else:
        log_test(f'Env var: {var_name}', 'FAIL', f'MISSING ({description})')
        has_required = False
        all_good = False

for var_name, description in optional_env_vars.items():
    if os.getenv(var_name):
        log_test(f'Env var: {var_name}', 'PASS')
    else:
        log_test(f'Env var: {var_name}', 'PASS', f'NOT SET ({description})')

# 6. Check voice processing setup
log_section('6. Checking Voice Processing Setup')
has_speech_key = bool(os.getenv('OPENAI_API_KEY') or os.getenv('GOOGLE_SPEECH_API_KEY') or os.getenv('GEMINI_API_KEY'))
has_translate_key = bool(os.getenv('GOOGLE_TRANSLATE_API_KEY') or os.getenv('GEMINI_API_KEY'))

if has_speech_key:
    log_test('Speech-to-Text API key', 'PASS', 'available')
else:
    log_test('Speech-to-Text API key', 'FAIL', 'missing')
    log('      Set OPENAI_API_KEY, GOOGLE_SPEECH_API_KEY, or GEMINI_API_KEY', Colors.YELLOW)
    all_good = False

if has_translate_key:
    log_test('Translation API key', 'PASS', 'available')
else:
    log_test('Translation API key', 'PASS', 'missing (will skip translation)')

# 7. Test imports
log_section('7. Testing Code Imports')
try:
    sys.path.insert(0, str(base_dir))
    import whatsapp
    log_test('whatsapp.py import', 'PASS')
except Exception as error:
    log_test('whatsapp.py import', 'FAIL', str(error))
    all_good = False

try:
    from services import voice_processor
    log_test('voice_processor.py import', 'PASS')
except Exception as error:
    log_test('voice_processor.py import', 'FAIL', str(error))
    all_good = False

try:
    import ai
    log_test('ai.py import', 'PASS')
except Exception as error:
    log_test('ai.py import', 'FAIL', str(error))
    all_good = False

try:
    import database
    log_test('database.py import', 'PASS')
except Exception as error:
    log_test('database.py import', 'FAIL', str(error))
    all_good = False

# 8. Check WhatsApp webhook configuration
log_section('8. Checking WhatsApp Configuration')
whatsapp_configured = bool(
    os.getenv('WHATSAPP_TOKEN') and
    os.getenv('WHATSAPP_PHONE_NUMBER_ID') and
    os.getenv('WHATSAPP_VERIFY_TOKEN')
)

if whatsapp_configured:
    log_test('WhatsApp webhook', 'PASS', 'configured')
else:
    log_test('WhatsApp webhook', 'FAIL', 'not fully configured')
    log('      Required: WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN', Colors.YELLOW)
    all_good = False

# Summary
print()
log('=' * 60, Colors.CYAN)
if all_good and has_required:
    log('✅ SETUP LOOKS GOOD! Ready to run the backend.', Colors.GREEN)
    print()
    log('📝 Next Steps:', Colors.BLUE)
    log('   1. Run: uvicorn app:app --reload --port 3001', Colors.BLUE)
    log('   2. Or use Docker: docker-compose up -d', Colors.BLUE)
    log('   3. Check health: curl http://localhost:3001/health', Colors.BLUE)
else:
    log('⚠️  SETUP INCOMPLETE - Please fix the issues above', Colors.YELLOW)
    print()
    log('💡 Quick Fixes:', Colors.BLUE)
    if not has_required:
        log('   - Add missing environment variables to .env file', Colors.BLUE)
    if missing_packages:
        log(f'   - Run: pip install {" ".join(missing_packages)}', Colors.BLUE)
log('=' * 60, Colors.CYAN)
print()

sys.exit(0 if (all_good and has_required) else 1)

