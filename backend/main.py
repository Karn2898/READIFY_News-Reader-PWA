from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import feedparser
from transformers import pipeline
import sqlite3
from TTS.api import TTS
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Free RSS feeds (local news)
RSS_FEEDS = [
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",  # TOI
    "https://rss.thehindu.com/dailies/topstories.rss",  # The Hindu
]

# Initialize models (cached after first run)
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

@app.get("/headlines")
async def get_headlines():
    articles = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:10]:
            # Simple headline filter (ML model)
            result = classifier(entry.title)
            if result[0]['label'] == 'POSITIVE' and result[0]['score'] > 0.7:
                # Summarize
                summary = summarizer(entry.summary[:500])[0]['summary_text']
                articles.append({
                    "title": entry.title,
                    "summary": summary,
                    "url": entry.link,
                    "published": str(entry.published)
                })
    return {"headlines": articles[:5]}

@app.get("/audio/{headline_id}")
async def get_audio(headline_id: int):
    # Generate TTS audio (saved locally)
    audio_path = f"audio_{headline_id}.wav"
    if not os.path.exists(audio_path):
        tts.tts_to_file(text="Sample news summary", file_path=audio_path)
    return {"audio_url": f"/static/{audio_path}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
