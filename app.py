"""
NexaBrain MVP Backend - FastAPI Application
Main application file with routes and middleware
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, Response, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv
import asyncio
import whatsapp
from database import get_conversations, get_recent_conversations

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    from services.voice_processor import voice_processor
    
    # Start cleanup task for processed messages
    whatsapp.start_cleanup_task()
    
    # Start voice processor cleanup task
    asyncio.create_task(voice_processor._cleanup_task())
    
    port = os.getenv('PORT', '3001')
    print(f'🚀 NexaBrain MVP Backend running on port {port}')
    print(f'📱 WhatsApp webhook: http://localhost:{port}/webhook')
    print(f'💻 API endpoint: http://localhost:{port}/conversations')
    print(f'🏥 Health check: http://localhost:{port}/health')
    print(f'📊 Webhook status: http://localhost:{port}/webhook/status')
    print(f'📚 API docs: http://localhost:{port}/docs')
    
    # Check environment variables
    webhook_status = whatsapp.get_webhook_status()
    if not webhook_status['configured']:
        print('⚠️  WhatsApp webhook not fully configured. Check your .env file.')
    
    yield
    
    # Shutdown (if needed)
    pass


# Initialize FastAPI app
app = FastAPI(
    title='NexaBrain MVP Backend',
    version='1.0.0',
    lifespan=lifespan
)

# CORS middleware
frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Request logging middleware
@app.middleware('http')
async def log_requests(request: Request, call_next) -> Response:
    """Log all incoming requests"""
    start_time = datetime.now()
    print(f'{request.method} {request.url.path}', {
        'timestamp': datetime.now().isoformat(),
        'client': request.client.host if request.client else 'unknown'
    })
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds() * 1000
    print(f'{request.method} {request.url.path} - {response.status_code} ({process_time:.0f}ms)')
    return response


# Routes
@app.get('/health')
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        webhook_status = whatsapp.get_webhook_status()
        return {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'webhook': webhook_status
        }
    except Exception as error:
        print(f'Error in health check: {error}')
        raise HTTPException(status_code=500, detail='Health check failed')


@app.get('/webhook/status')
async def webhook_status() -> Dict[str, Any]:
    """Get webhook configuration status"""
    try:
        status = whatsapp.get_webhook_status()
        return status
    except Exception as error:
        print(f'Error getting webhook status: {error}')
        raise HTTPException(status_code=500, detail='Failed to get webhook status')


@app.get('/webhook')
async def verify_webhook(
    mode: str = Query(..., alias='hub.mode'),
    token: str = Query(..., alias='hub.verify_token'),
    challenge: str = Query(..., alias='hub.challenge')
) -> PlainTextResponse:
    """Verify webhook (GET request from WhatsApp)"""
    try:
        result = await whatsapp.verify_webhook(mode, token, challenge)
        if result:
            return PlainTextResponse(result)
        return PlainTextResponse('Forbidden', status_code=403)
    except Exception as error:
        print(f'Error in webhook verification: {error}')
        return PlainTextResponse('Internal server error', status_code=500)


@app.post('/webhook')
async def handle_webhook(
    request: Request,
    background_tasks: BackgroundTasks
) -> PlainTextResponse:
    """Handle incoming messages (POST request from WhatsApp)"""
    try:
        # Quick acknowledgment to WhatsApp (within 20 seconds)
        body = await request.json()
        
        # Process message in background
        background_tasks.add_task(whatsapp.handle_message, body)
        
        return PlainTextResponse('EVENT_RECEIVED', status_code=200)
    except Exception as error:
        print(f'Error handling webhook: {error}')
        # Always acknowledge to prevent WhatsApp retries
        return PlainTextResponse('EVENT_RECEIVED', status_code=200)


@app.get('/conversations')
async def get_conversations_endpoint(
    limit: Optional[int] = Query(None, ge=1, le=1000),
    phone: Optional[str] = Query(None)
) -> List[Dict[str, Any]]:
    """Get all conversations with optional filters"""
    try:
        conversations = await get_conversations()
        
        # Filter by phone if provided
        if phone:
            conversations = [c for c in conversations if c.get('phone') == phone]
        
        # Apply limit if provided
        if limit:
            conversations = conversations[:limit]
        
        return conversations
    except Exception as error:
        print(f'Error fetching conversations: {error}')
        raise HTTPException(status_code=500, detail='Failed to fetch conversations')


@app.get('/conversations/{phone}')
async def get_user_conversations(phone: str) -> List[Dict[str, Any]]:
    """Get conversations for a specific phone number"""
    try:
        conversations = await get_recent_conversations(phone, 50)
        return conversations
    except Exception as error:
        print(f'Error fetching user conversations: {error}')
        raise HTTPException(status_code=500, detail='Failed to fetch user conversations')


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={'error': 'Route not found'}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={'error': 'Internal server error'}
    )


if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 3001))
    uvicorn.run(app, host='0.0.0.0', port=port)

