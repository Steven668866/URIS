# Real-time Voice Input Setup Guide

## Overview

The URIS application now supports **real-time continuous speech-to-text** instead of the previous click-to-record method. This provides a more natural conversational experience with live transcription.

## Key Features

- 🎤 **Continuous Speech Recognition**: Speak naturally without clicking record/stop buttons
- 🔄 **Live Transcription**: See text appear in real-time as you speak
- 📝 **Smart Text Handling**: Distinguishes between partial (live) and final (confirmed) transcripts
- 🌐 **WebSocket Integration**: Connects to external STT services for high-quality recognition

## Setup Requirements

### 1. Install Dependencies

```bash
pip install websocket-client>=1.6.0 numpy>=1.21.0
```

Or uncomment the lines in `requirements.txt`:
```txt
websocket-client>=1.6.0
numpy>=1.21.0
```

### 2. Set Up STT WebSocket Server

You need a WebSocket server that provides real-time speech-to-text. The app expects:

- **URL**: `ws://localhost:8765` (configurable in `STT_CONFIG`)
- **Audio Format**: 16kHz, mono PCM
- **Protocol**: JSON messages

#### Example Server Response Format

```json
// Partial transcript (live text)
{
  "type": "transcript",
  "transcript": "Hello, I would like to",
  "is_final": false
}

// Final transcript (confirmed sentence)
{
  "type": "transcript",
  "transcript": "Hello, I would like to ask about this video.",
  "is_final": true
}
```

### 3. Recommended STT Services

#### AssemblyAI Streaming API
```python
# Example implementation
import assemblyai as aai
from assemblyai import StreamingClient, StreamingClientOptions, StreamingParameters

def on_turn(event):
    # Send transcript to WebSocket clients
    data = {
        "type": "transcript",
        "transcript": event.transcript,
        "is_final": event.is_final
    }
    # Send to connected WebSocket clients

client = StreamingClient(StreamingClientOptions(api_key="your-api-key"))
client.connect(StreamingParameters(sample_rate=16000))
client.stream(aai.extras.MicrophoneStream(sample_rate=16000))
```

#### Alternative: Local Whisper Integration
You can also integrate with local Whisper models for offline processing.

## Usage

### 1. Start Voice Input
- Click **"🎙️ Start Voice Input"** in the sidebar
- The system connects to your STT WebSocket server
- Status shows "🎤 Voice input active - Speak continuously"

### 2. Speak Naturally
- Begin speaking immediately after starting
- Text appears in real-time in the **"🔄 Live Transcript"** section
- Confirmed sentences move to **"📝 Final Transcript"**

### 3. Send to Chat
- **Option 1**: Click **"📤 Send to Chat"** to use current transcript
- **Option 2**: Stop voice input, and the final transcript becomes available for chat

### 4. Stop Voice Input
- Click **"⏹️ Stop Voice Input"** to end the session
- Final transcript remains available until cleared

## Configuration

Modify `STT_CONFIG` in `app.py` to customize:

```python
STT_CONFIG = {
    "websocket_url": "ws://localhost:8765",  # Your STT server URL
    "sample_rate": 16000,                    # Audio sample rate
    "channels": 1,                           # Mono audio
    "chunk_size": 1024,                      # Audio chunk size
    "buffer_duration": 0.1                   # Buffer duration (100ms)
}
```

## Troubleshooting

### Connection Issues
- Ensure your STT WebSocket server is running on the configured URL
- Check firewall settings for WebSocket connections
- Verify server implements the expected JSON protocol

### Audio Quality Issues
- Ensure microphone has proper permissions
- Check audio levels and background noise
- Verify sample rate matches server expectations (16kHz recommended)

### Performance Issues
- Reduce `buffer_duration` for lower latency (but may affect stability)
- Increase `chunk_size` for better throughput
- Monitor WebSocket connection stability

## Migration from Legacy Voice Input

If you were using the old `streamlit-mic-recorder` method:

1. **Remove old dependency**: `pip uninstall streamlit-mic-recorder`
2. **Install new dependencies**: `pip install websocket-client numpy`
3. **Set up STT server**: Implement or connect to a WebSocket-based STT service
4. **Update workflow**: No more clicking record/stop - just continuous conversation!

## Example Implementation

See the voice processing worker function in `app.py` for the client-side WebSocket handling logic. The server-side implementation will depend on your chosen STT service.
