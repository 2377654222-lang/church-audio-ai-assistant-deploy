# Church Audio AI Assistant

Flat Render deployment version.

Upload these root files to GitHub:

- `server.py`
- `index.html`
- `requirements.txt`
- `Dockerfile`
- `render.yaml`
- `README.md`

Then deploy on Render with:

```text
New + -> Blueprint
```

The app provides:

- React frontend from `index.html`
- FastAPI backend from `server.py`
- PostgreSQL through Render
- Audio upload
- librosa feature extraction
- OpenAI analysis when `OPENAI_API_KEY` is set
- Free rule-based reports when no API key is set

Default Render settings:

- Maximum upload: 75 MB
- Analysis window: first 60 seconds
- Analysis sample rate: 22050 Hz

The analysis window keeps the free Render instance stable for longer recordings.
