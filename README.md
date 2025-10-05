# HebTTS Backend

RunPod serverless wrapper for [HebTTS](https://github.com/slp-rl/HebTTS) Hebrew text-to-speech model with **intelligent text chunking** for long texts.

## Features

- ✅ **Smart text chunking** - Handles long texts automatically (breaks at Hebrew sentence/clause boundaries)
- ✅ **Backward compatible** - All existing API calls work unchanged
- ✅ **Configurable** - Adjust chunking behavior or disable it
- ✅ **Seamless audio** - Concatenates chunks for smooth output

## API

**RunPod Job Input:**
```json
{
  "input": {
    "text": "Hebrew text here (any length)",
    "speaker": "osim",            // or "geek", "shaul"
    "top_k": 15,                  // optional, default 15
    "temperature": 0.6,           // optional, default 0.6
    "use_mbd": true,              // optional, default true (higher quality)
    "enable_chunking": true,      // optional, default true (auto-chunk long texts)
    "max_chunk_chars": 150        // optional, default 150 (chars per chunk)
  }
}
```

**Returns:** Base64 audio + metadata
```json
{
  "audio_base64": "...",
  "filename": "output.wav", 
  "sample_rate": 24000,
  "chunked": true,               // true if text was chunked
  "chunks_processed": 3,         // number of chunks (if chunked)
  "original_length": 425         // original text length
}
```

## Chunking Logic

- **Short texts (≤150 chars):** Single chunk processing
- **Long texts (>150 chars):** Auto-chunked at Hebrew boundaries (sentences → clauses → words)
- **Token limit:** Keeps under 512 token model limit
- **Quality:** Smart boundary detection preserves natural speech flow

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Download model checkpoint (~2GB) to HebTTSLM/checkpoint.pt
# Test with Docker
docker-compose up
```

## Deployment

Deploy to **RunPod Serverless** with existing Docker configuration. No changes needed - chunking works automatically.