#!/usr/bin/env python3
"""
NexaBrain MVP Workflow Test Script

Tests the entire backend workflow:
1. Environment variables
2. Backend health check
3. API endpoints
4. AI integration
5. Database operations
6. Error handling
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Import modules
try:
    from ai import generate_ai_response, test_gemini_connection
    from database import save_conversation, get_conversations, get_recent_conversations
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

# Configuration
API_BASE_URL = os.getenv('TEST_API_URL', 'http://localhost:3001')

# Colors
class Colors:
    RESET = '\033[0m'
    GREEN = '\033[32m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'

# Test results
passed_tests = 0
failed_tests = 0
test_results: List[Dict[str, Any]] = []

def log(message: str, color: str = Colors.RESET) -> None:
    """Print colored message"""
    print(f"{color}{message}{Colors.RESET}")

def log_test(name: str, status: str, message: str = '') -> None:
    """Log test result"""
    icon = '✅' if status == 'PASS' else '❌' if status == 'FAIL' else '⚠️'
    color = Colors.GREEN if status == 'PASS' else Colors.RED if status == 'FAIL' else Colors.YELLOW
    log(f"{icon} {name}{': ' + message if message else ''}", color)
    
    test_results.append({'name': name, 'status': status, 'message': message})
    global passed_tests, failed_tests
    if status == 'PASS':
        passed_tests += 1
    elif status == 'FAIL':
        failed_tests += 1

def log_section(title: str) -> None:
    """Log section header"""
    print()
    log('=' * 60, Colors.CYAN)
    log(f"  {title}", Colors.CYAN)
    log('=' * 60, Colors.CYAN)
    print()

# Test functions
async def test_environment_variables() -> bool:
    """Test environment variables"""
    log_section('1. Environment Variables Check')
    
    required_vars: List[str] = [
        'GEMINI_API_KEY',
        'WHATSAPP_TOKEN',
        'WHATSAPP_PHONE_NUMBER_ID',
        'WHATSAPP_VERIFY_TOKEN'
    ]

    all_present = True
    for var_name in required_vars:
        if os.getenv(var_name):
            log_test(f'Env var: {var_name}', 'PASS')
        else:
            log_test(f'Env var: {var_name}', 'FAIL', 'Missing')
            all_present = False

    return all_present

async def test_backend_health() -> bool:
    """Test backend health endpoint"""
    log_section('2. Backend Health Check')
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f'{API_BASE_URL}/health')
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok':
                    log_test('Backend health endpoint', 'PASS')
                    log(f"   Status: {data.get('status')}", Colors.BLUE)
                    log(f"   Webhook configured: {data.get('webhook', {}).get('configured', False)}", Colors.BLUE)
                    return True
                else:
                    log_test('Backend health endpoint', 'FAIL', 'Unexpected response')
                    return False
            else:
                log_test('Backend health endpoint', 'FAIL', f'Status code: {response.status_code}')
                return False
    except httpx.ConnectError:
        log_test('Backend health endpoint', 'FAIL', 'Cannot connect - is backend running?')
        log('   Start backend: uvicorn app:app --reload --port 3001', Colors.YELLOW)
        return False
    except Exception as error:
        log_test('Backend health endpoint', 'FAIL', str(error))
        return False

async def test_webhook_status() -> bool:
    """Test webhook status endpoint"""
    log_section('3. Webhook Status Check')
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f'{API_BASE_URL}/webhook/status')
            
            if response.status_code == 200:
                data = response.json()
                log_test('Webhook status endpoint', 'PASS')
                log(f"   Configured: {data.get('configured', False)}", Colors.BLUE)
                log(f"   Has token: {data.get('hasToken', False)}", Colors.BLUE)
                log(f"   Has phone number ID: {data.get('hasPhoneNumberId', False)}", Colors.BLUE)
                return True
            else:
                log_test('Webhook status endpoint', 'FAIL', f'Status code: {response.status_code}')
                return False
    except Exception as error:
        log_test('Webhook status endpoint', 'FAIL', str(error))
        return False

async def test_api_endpoints() -> bool:
    """Test API endpoints"""
    log_section('4. API Endpoints Check')
    
    try:
        # Test /conversations endpoint
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f'{API_BASE_URL}/conversations')
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    log_test('/conversations endpoint', 'PASS', f'{len(data)} conversations')
                else:
                    log_test('/conversations endpoint', 'FAIL', 'Invalid response format')
                    return False
            else:
                log_test('/conversations endpoint', 'FAIL', f'Status code: {response.status_code}')
                return False

            # Test /conversations/:phone endpoint
            response = await client.get(f'{API_BASE_URL}/conversations/919876543210')
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    log_test('/conversations/:phone endpoint', 'PASS')
                else:
                    log_test('/conversations/:phone endpoint', 'FAIL', 'Invalid response format')
                    return False
            else:
                log_test('/conversations/:phone endpoint', 'FAIL', f'Status code: {response.status_code}')
                return False

        return True
    except Exception as error:
        log_test('API endpoints', 'FAIL', str(error))
        return False

async def test_ai_integration() -> bool:
    """Test AI integration"""
    log_section('5. AI Integration Check')
    
    try:
        # Test Gemini connection
        result = await test_gemini_connection()
        
        if result.get('success'):
            log_test('Gemini AI connection', 'PASS', result.get('response', ''))
        else:
            log_test('Gemini AI connection', 'FAIL', result.get('error', 'Unknown error'))
            return False

        # Test AI response generation
        test_message = "Hello, how are you?"
        response = await generate_ai_response(test_message, [], 'TestUser', 'en')
        
        if response and len(response) > 0:
            log_test('AI response generation', 'PASS', f'Response: {response[:50]}...')
        else:
            log_test('AI response generation', 'FAIL', 'Empty response')
            return False

        return True
    except Exception as error:
        log_test('AI integration', 'FAIL', str(error))
        return False

async def test_database_operations() -> bool:
    """Test database operations"""
    log_section('6. Database Operations Check')
    
    try:
        # Test get conversations
        conversations = await get_conversations()
        log_test('Get conversations', 'PASS', f'{len(conversations)} found')
        
        # Test get recent conversations
        recent = await get_recent_conversations('919876543210', 5)
        log_test('Get recent conversations', 'PASS', f'{len(recent)} found')
        
        # Test save conversation
        test_conv = {
            'phone': '919876543210',
            'userName': 'TestUser',
            'userMessage': 'Test message',
            'aiResponse': 'Test response',
            'timestamp': '2024-01-01T00:00:00Z'
        }
        saved = await save_conversation(test_conv)
        
        if saved:
            log_test('Save conversation', 'PASS', f'ID: {saved.get("id", "unknown")}')
        else:
            log_test('Save conversation', 'FAIL', 'Failed to save')
            return False

        return True
    except Exception as error:
        log_test('Database operations', 'FAIL', str(error))
        return False

async def test_error_handling() -> bool:
    """Test error handling"""
    log_section('7. Error Handling Check')
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Test 404 endpoint
            response = await client.get(f'{API_BASE_URL}/nonexistent')
            
            if response.status_code == 404:
                log_test('404 error handling', 'PASS')
            else:
                log_test('404 error handling', 'FAIL', f'Expected 404, got {response.status_code}')
                return False

        return True
    except Exception as error:
        log_test('Error handling', 'FAIL', str(error))
        return False

# Main test runner
async def run_all_tests() -> int:
    """Run all tests"""
    print()
    log('🧪 NexaBrain MVP Workflow Test Suite', Colors.CYAN)
    log('=' * 60, Colors.CYAN)
    print()
    
    results: List[bool] = []
    results.append(await test_environment_variables())
    results.append(await test_backend_health())
    results.append(await test_webhook_status())
    results.append(await test_api_endpoints())
    results.append(await test_ai_integration())
    results.append(await test_database_operations())
    results.append(await test_error_handling())
    
    # Summary
    print()
    log('=' * 60, Colors.CYAN)
    log('📊 Test Summary', Colors.CYAN)
    log('=' * 60, Colors.CYAN)
    print()
    log(f'✅ Passed: {passed_tests}', Colors.GREEN)
    log(f'❌ Failed: {failed_tests}', Colors.RED)
    log(f'📈 Total: {passed_tests + failed_tests}', Colors.BLUE)
    print()
    
    if all(results):
        log('✅ ALL TESTS PASSED!', Colors.GREEN)
        return 0
    else:
        log('⚠️  SOME TESTS FAILED', Colors.YELLOW)
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

