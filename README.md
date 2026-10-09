DM# Local AI Memory Engine (`mem-engine`)

A privacy-first, locally deployed long-term memory engine for AI chatbots. It extracts facts, preferences, goals, and project details from natural language, dynamically resolves contradictions, preserves revision history, and retrieves relevant memories via hybrid semantic + keyword search.

## Setup & Execution Guide

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start live server with hot-reload
uvicorn main:app --reload --port 8000