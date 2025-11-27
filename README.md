# SmartVisionQA: Real-Time Video Understanding & Query System

An AI-powered system that enables users to upload or stream video content and ask natural language questions about events happening within the video using state-of-the-art multimodal LLMs and computer vision.

## 🎯 Real-World Use Cases

- **Security Teams**: Query surveillance footage for specific events without manual review
- **Teachers**: Review classroom recordings and identify key moments
- **Content Reviewers**: Quickly analyze video content at scale
- **Sports Analysts**: Find specific plays and events in game footage

## ✨ Advanced Features

- 🎥 **Real-time Video Frame Processing**: Automatic chunking and CV model analysis
- 💬 **Multimodal Q&A**: Ask questions like "Who entered the room after the teacher?"
- 🔍 **Semantic Video Search**: Find moments like "3 people standing near the whiteboard"
- 📊 **Event Timeline Generation**: AI-powered summarization and clustering
- 🔒 **Video Redaction**: Automatic face/object blurring
- 👥 **Collaborative Annotation**: Real-time collaborative mode with WebSocket

## 🛠️ Tech Stack

### Frontend
- React with Vite
- Tailwind CSS
- WebRTC for streaming
- Socket.IO for real-time collaboration

### Backend
- FastAPI (Python)
- OpenCV for video processing
- YOLOv8 for object detection
- BLIP/LLaVA for vision-language reasoning
- Whisper for audio transcription
- ChromaDB for vector embeddings

### Infrastructure
- Docker & Docker Compose
- Redis for task queuing
- Local file storage (can extend to AWS S3)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/shreyaupretyy/smart-vision-qa.git
cd smart-vision-qa
```

2. **Set up backend**
```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Set up frontend**
```bash
cd frontend
npm install
cd ..
```

4. **Configure environment**
```bash
# Copy example env file
copy .env.example .env  # Windows
# or
cp .env.example .env    # Linux/Mac

# Edit .env with your settings
```

5. **Run with Docker (Recommended)**
```bash
docker-compose up --build
```

Or run manually:

```bash
# Terminal 1: Backend
venv\Scripts\activate  # Windows
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

6. **Access the application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📁 Project Structure

```
SmartVisionQA/
├── backend/
│   ├── main.py                 # FastAPI application entry
│   ├── api/
│   │   ├── routes.py          # API endpoints
│   │   └── websocket.py       # WebSocket handlers
│   ├── core/
│   │   ├── config.py          # Configuration
│   │   └── dependencies.py    # Dependency injection
│   ├── services/
│   │   ├── video_processor.py # Video processing
│   │   ├── object_detection.py# YOLOv8 integration
│   │   ├── vision_qa.py       # Vision-language QA
│   │   ├── transcription.py   # Whisper integration
│   │   ├── embeddings.py      # Vector embeddings
│   │   ├── timeline.py        # Event timeline
│   │   └── redaction.py       # Video redaction
│   └── models/
│       └── schemas.py         # Pydantic models
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API services
│   │   └── utils/             # Utilities
│   ├── public/
│   └── package.json
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🔧 API Endpoints

### Video Management
- `POST /api/video/upload` - Upload video file
- `GET /api/video/{video_id}` - Get video metadata
- `DELETE /api/video/{video_id}` - Delete video

### Analysis
- `POST /api/analyze/query` - Ask questions about video
- `POST /api/analyze/detect` - Run object detection
- `POST /api/analyze/transcribe` - Transcribe audio
- `GET /api/analyze/timeline/{video_id}` - Get event timeline
- `POST /api/analyze/search` - Semantic search in video

### Redaction
- `POST /api/redact/faces` - Blur faces in video
- `POST /api/redact/objects` - Blur specific objects

### Real-time
- `WebSocket /ws/stream` - Real-time video streaming
- `WebSocket /ws/collaborate` - Collaborative annotation

## 🎨 Features in Detail

### 1. Video Upload & Processing
Upload videos in various formats (MP4, AVI, MOV, MKV). The system automatically extracts frames, generates embeddings, and prepares the video for analysis.

### 2. Multimodal Question Answering
Ask natural language questions about video content:
- "What objects are visible in the first 30 seconds?"
- "Who is wearing a red shirt?"
- "When did the person leave the room?"

### 3. Object Detection & Tracking
Real-time object detection using YOLOv8 with support for 80+ object classes. Track objects across frames for temporal analysis.

### 4. Semantic Search
Search for specific moments using natural language:
- "Find scenes with multiple people"
- "Show me when the door opens"
- "Locate moments with vehicles"

### 5. Event Timeline
Automatically generate a timeline of key events with AI-powered summarization and clustering of similar moments.

### 6. Video Redaction
Automatically detect and blur faces or specific objects for privacy protection. Export redacted videos with one click.

### 7. Collaborative Annotation
Multiple users can annotate videos in real-time with synchronized cursors and annotations.

## 🧪 Development

### Running Tests
```bash
# Backend tests
pytest backend/tests/

# Frontend tests
cd frontend
npm test
```

### Code Quality
```bash
# Format code
black backend/
ruff check backend/

# Frontend linting
cd frontend
npm run lint
```

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- OpenAI Whisper
- Salesforce BLIP
- ChromaDB
- FastAPI
- React & Vite

---

Built with ❤️ by the SmartVisionQA team
