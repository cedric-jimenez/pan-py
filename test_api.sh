#!/bin/bash

echo "🧪 Testing Salamander Detection API"
echo "===================================="

# Check if server is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Server not running on http://localhost:8000"
    echo "   Start it with: make run"
    exit 1
fi

# Test 1: Health check
echo -e "\n1️⃣  Testing /health endpoint..."
curl -s http://localhost:8000/health | jq '.'

# Test 2: Root endpoint
echo -e "\n2️⃣  Testing / endpoint..."
curl -s http://localhost:8000/ | jq '.'

# Test 3: Model info
echo -e "\n3️⃣  Testing /model-info endpoint..."
curl -s http://localhost:8000/model-info | jq '.'

# Test 4: Detection (if image exists)
if [ -f "imagestest/salamander.jpg" ]; then
    echo -e "\n4️⃣  Testing /crop-salamander endpoint..."
    curl -s -X POST "http://localhost:8000/crop-salamander?return_base64=false" \
      -F "file=@imagestest/salamander.jpg" | jq '.'
else
    echo -e "\n⚠️  No test image found at imagestest/salamander.jpg"
    echo "   Please add a test image to run detection tests"
fi

# Test 5: Segmentation (if image exists)
if [ -f "imagestest/salamander.jpg" ]; then
    echo -e "\n5️⃣  Testing /segment-salamander endpoint..."
    curl -s -X POST "http://localhost:8000/segment-salamander?return_base64=false&image_quality=85&max_size=640" \
      -F "file=@imagestest/salamander.jpg" | jq '.'
fi

echo -e "\n✅ Tests completed!"
