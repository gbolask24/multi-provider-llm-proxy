#!/bin/bash
# Sample requests for the Multi-Provider LLM Proxy
# Usage: bash scripts/sample_requests.sh
# Requires: the service running on localhost:8000

BASE_URL="http://localhost:8000"

echo "=== Health Check ==="
curl -s "$BASE_URL/health" | python -m json.tool
echo ""

echo "=== List Providers ==="
curl -s "$BASE_URL/v1/providers" | python -m json.tool
echo ""

echo "=== Basic OpenAI Request ==="
curl -s -X POST "$BASE_URL/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain what an LLM proxy is in one sentence.",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "max_tokens": 100
  }' | python -m json.tool
echo ""

echo "=== Request with Fallback Chain ==="
curl -s -X POST "$BASE_URL/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "fallback_chain": [
      {"provider": "anthropic", "model": "claude-3-haiku-20240307"}
    ],
    "metadata": {
      "use_case": "testing",
      "environment": "local"
    }
  }' | python -m json.tool
echo ""

echo "=== Request with System Prompt and Messages ==="
curl -s -X POST "$BASE_URL/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Summarise this in a professional tone: The server crashed again last night."}
    ],
    "system_prompt": "You are a professional technical writer.",
    "provider": "anthropic",
    "model": "claude-3-haiku-20240307",
    "temperature": 0.2,
    "max_tokens": 200
  }' | python -m json.tool
echo ""

echo "=== Unsupported Provider (expect error) ==="
curl -s -X POST "$BASE_URL/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "provider": "nonexistent",
    "model": "some-model"
  }' | python -m json.tool
echo ""
