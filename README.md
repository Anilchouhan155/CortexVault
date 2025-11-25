# NexaBrain MVP - Python Backend

Python 3 backend for NexaBrain MVP using FastAPI. This is a rewrite of the Node.js backend with the same functionality.

## Features

- **WhatsApp Webhook Integration** - Receive and send messages via WhatsApp Business API
- **AI Responses** - Google Gemini AI with friend-first personality
- **Voice Processing** - Speech-to-text (Whisper) and text-to-speech support
- **Database** - Supabase (Postgres) with JSON file fallback
- **Multi-language** - Support for English and Hindi

## Requirements

- Python 3.8 or higher (or Docker)
- pip (Python package manager)
- Docker (optional, for containerized deployment)

## Setup

### 1. Install Dependencies

```bash
cd nexabrain_py
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the `nexabrain_py` directory:

```env
# WhatsApp Business API Configuration
WHATSAPP_TOKEN=your_whatsapp_token_here
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_VERIFY_TOKEN=your_verify_token
WHATSAPP_URL=https://graph.facebook.com/v18.0

# Google Gemini AI
GEMINI_API_KEY=your_gemini_api_key

# Supabase Database (Optional - falls back to JSON if not set)
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# OpenAI (for Whisper STT and TTS)
OPENAI_API_KEY=your_openai_api_key

# Google Cloud APIs (Optional - fallback for STT/TTS)
GOOGLE_SPEECH_API_KEY=your_google_speech_api_key
GOOGLE_TRANSLATE_API_KEY=your_google_translate_api_key

# Server Configuration
PORT=3001
FRONTEND_URL=http://localhost:3000
NODE_ENV=development
```

### 3. Run the Server

**Option A: Direct Python (Development)**
```bash
# Development mode with auto-reload
uvicorn app:app --reload --port 3001

# Or use Python directly
python app.py
```

**Option B: Docker (Production)**
```bash
# Build and run with Docker
docker build -t nexabrain-py .
docker run -p 3001:3001 --env-file .env nexabrain-py

# Or use docker-compose (recommended)
docker-compose up -d
```

## API Endpoints

- `GET /health` - Health check with webhook status
- `GET /webhook/status` - Webhook configuration status
- `GET /webhook` - Webhook verification (WhatsApp)
- `POST /webhook` - Handle incoming messages (WhatsApp)
- `GET /conversations` - Get all conversations (with optional `limit` and `phone` query params)
- `GET /conversations/{phone}` - Get conversations for a specific phone number
- `GET /docs` - Interactive API documentation (Swagger UI)

## Project Structure

```
nexabrain_py/
├── app.py                 # Main FastAPI application
├── whatsapp.py            # WhatsApp webhook handler
├── ai.py                  # Gemini AI integration
├── database.py            # Supabase + JSON storage
├── services/
│   └── voice_processor.py # Voice STT/TTS processing
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── .dockerignore          # Docker ignore patterns
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Key Differences from Node.js Version

- Uses **FastAPI** instead of Express.js
- All operations are **async/await** for better performance
- Uses **httpx** for async HTTP requests
- Uses **pathlib** for file operations
- Automatic API documentation at `/docs`

## Docker

### Building the Docker Image

```bash
docker build -t nexabrain-py .
```

### Running with Docker

```bash
# Run with environment variables from .env file
docker run -p 3001:3001 --env-file .env nexabrain-py

# Or run with docker-compose (recommended)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Docker Compose

The `docker-compose.yml` file includes:
- Automatic environment variable loading from `.env`
- Volume mounts for temp files and conversations.json
- Health checks
- Auto-restart on failure

## Development

### Running in Development Mode

```bash
# With virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

uvicorn app:app --reload --host 0.0.0.0 --port 3001
```

### Testing Webhook Locally

Use ngrok to expose your local server:

```bash
ngrok http 3001
```

Then configure the webhook URL in WhatsApp Business API dashboard.

## Notes

- The backend maintains the same API contract as the Node.js version
- All endpoints return the same response formats
- Error handling matches the Node.js implementation
- Voice processing requires OpenAI API key (for Whisper) or Google Speech API key

