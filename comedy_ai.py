import openai
import os
from dotenv import load_dotenv
import whisper
import librosa
import ssl

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Handle SSL certificate errors for Whisper
ssl._create_default_https_context = ssl._create_unverified_context

# Load Whisper model once
whisper_model = whisper.load_model("base")

# Joke feedback function (originally missing!)
def joke_feedback(joke_text):
    prompt = f"""
    Analyze this joke for humor, structure, and clarity. Provide concise feedback and suggest improvements:

    "{joke_text}"
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a supportive comedy coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# Audio transcription function
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Audio metrics analysis function
def analyze_audio_metrics(audio_path):
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    speaking_rate = len(y) / duration
    pauses = librosa.effects.split(y, top_db=25)
    num_pauses = len(pauses) - 1
    avg_loudness = librosa.feature.rms(y=y).mean()
    
    return {
        "duration_seconds": duration,
        "speaking_rate": speaking_rate,
        "num_pauses": num_pauses,
        "avg_loudness": avg_loudness
    }
