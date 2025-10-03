#!/bin/bash

# Test script for HebTTS API
API_URL="http://localhost:5000"

echo "🔍 Testing HebTTS API..."
echo "=========================="

# Test 1: Health check
echo "1. Testing health endpoint..."
curl -s "${API_URL}/health" | jq '.' || echo "❌ Health check failed"
echo ""

# Test 2: Get speakers
echo "2. Testing speakers endpoint..."
curl -s "${API_URL}/speakers" | jq '.' || echo "❌ Speakers check failed"
echo ""

# Test 3: Synthesize text
echo "3. Testing synthesis (this will download a WAV file)..."
curl -X POST "${API_URL}/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "שלום עולם, זהו בדיקה",
    "speaker": "osim",
    "filename": "test_output"
  }' \
  --output test_output.wav \
  --write-out "HTTP Status: %{http_code}\nDownload size: %{size_download} bytes\n"

# Check if file was created
if [ -f "test_output.wav" ]; then
    echo "✅ WAV file created successfully!"
    echo "📁 File size: $(du -h test_output.wav | cut -f1)"
    echo "🎵 You can play it with: aplay test_output.wav (or any audio player)"
else
    echo "❌ WAV file was not created"
fi

echo ""
echo "🏁 Test complete!"