# HebTTS Backend

Flask API wrapper for [HebTTS](https://github.com/slp-rl/HebTTS) Hebrew text-to-speech model.

## Local Testing (Powerful PC Required)

**Setup:**

```bash
# Install dependencies
pip install -r requirements.txt

# Download the model checkpoint (~2GB)
cd HebTTSLM
# Get checkpoint.pt from the HebTTS repo/releases
# Place it in HebTTSLM/checkpoint.pt

# Run the server
cd ..
python app.py
```

Server runs at `http://localhost:5000`

**Test it:**

```bash
./test_api.sh
# Or manually:
curl -X POST http://localhost:5000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "שלום עולם", "speaker": "osim"}' \
  --output test.wav
```

## API

**POST /synthesize**
```json
{
  "text": "Hebrew text here",
  "speaker": "osim",  // or "geek", "shaul"
  "top_k": 15,        // optional, default 15
  "temperature": 0.6, // optional, default 0.6
  "use_mbd": true     // optional, default true (higher quality)
}
```

Returns: WAV file (24kHz)

**GET /health** - Check if server is running

**GET /speakers** - List available speakers

## Deployment

The ideal deployment solution would be a pay-as-you-go service such as RunPod, which offer powerful GPUs. 
Simply open an account, setup a new serverless endpoint, and deploy.