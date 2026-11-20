# YouTube Shorts Maker

Streamlit app for producing 30-second, 9:16 clips from YouTube videos, with audio controls, optional voiceover/music, Whisper captions, and automated exposed-content screening.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, this repository, branch **main**, and **app.py**.
4. Deploy. Streamlit installs packages from `requirements.txt` and system packages from `packages.txt`.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Important

- Only use source videos you own or have permission to reuse.
- Moderation screening is automated and can miss content or flag content incorrectly.
- Whisper and NudeNet are resource-intensive. Streamlit Community Cloud resource limits may affect processing.
- The app is intended for short processing jobs. Longer source videos may fail due to memory/time limits.
