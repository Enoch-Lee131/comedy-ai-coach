import os
import openai
import whisper
import librosa
import ssl
import torchaudio
import torchaudio.transforms as transforms
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Handle SSL certificate errors for Whisper
ssl._create_default_https_context = ssl._create_unverified_context

# Load Whisper model once for performance
whisper_model = whisper.load_model("base")

# Get OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def joke_feedback(joke_text):
    """Analyzes a joke using OpenAI and provides feedback on humor, structure, and clarity."""
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

def convert_mp3_to_wav(input_path, output_path="converted_audio.wav"):
    """Converts MP3 files to WAV format using torchaudio."""
    try:
        waveform, sample_rate = torchaudio.load(input_path)
        torchaudio.save(output_path, waveform, sample_rate, format="wav")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Audio conversion error: {str(e)}. Ensure the uploaded file is a valid MP3.")

def transcribe_audio(audio_path):
    """Transcribe audio file using OpenAI Whisper."""
    if audio_path.endswith(".mp3"):
        audio_path = convert_mp3_to_wav(audio_path)  # Convert MP3 to WAV before transcription

    result = whisper_model.transcribe(audio_path)
    return result["text"]

def analyze_audio_metrics(audio_path):
    """Extracts audio metrics such as duration, speaking rate, number of pauses, and loudness."""
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
