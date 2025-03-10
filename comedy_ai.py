import os
import openai
import whisper
import librosa
import ssl
from dotenv import load_dotenv
import torchaudio
import numpy as np

# --- Monkey-patch Whisper's audio loader to use torchaudio instead of FFmpeg ---

def load_audio_torchaudio(filename, sr=16000):
    """
    Load an audio file using torchaudio, resample to the target sampling rate,
    convert to mono, and normalize to the [-1, 1] range.
    """
    # Load the audio file (supports MP3, WAV, etc.)
    waveform, orig_sr = torchaudio.load(filename)
    
    # Resample if needed
    if orig_sr != sr:
        transform = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = transform(waveform)
    
    # Convert to mono by averaging channels if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    
    audio = waveform.squeeze().numpy().astype(np.float32)
    
    # Normalize to [-1, 1]
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio

# Replace Whisper's load_audio with our torchaudio-based version.
import whisper.audio
whisper.audio.load_audio = load_audio_torchaudio

# --- End monkey-patching ---

# Load environment variables from .env file
load_dotenv()

# Handle SSL certificate errors for Whisper
ssl._create_default_https_context = ssl._create_unverified_context

# Load the Whisper model (this downloads the model if not present)
whisper_model = whisper.load_model("base")

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def joke_feedback(joke_text):
    """
    Uses OpenAI's GPT-3.5-turbo to analyze a joke for humor, structure, and clarity.
    Returns constructive feedback and suggestions.
    """
    prompt = f"""
    You are a supportive comedy coach. Analyze this joke for humor, structure, and clarity.
    Provide constructive feedback and specific suggestions for improvement.

    Joke:
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

def transcribe_audio(audio_path):
    """
    Transcribes the given audio file using the Whisper model.
    The monkey-patched load_audio function will load the file using torchaudio.
    """
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def analyze_audio_metrics(audio_path):
    """
    Analyzes the audio file to extract basic metrics: duration, speaking rate,
    number of pauses, and average loudness.
    """
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
