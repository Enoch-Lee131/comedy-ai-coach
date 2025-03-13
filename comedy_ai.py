import os
import openai
import whisper
import librosa
import ssl
from dotenv import load_dotenv
import torchaudio
import imageio_ffmpeg

load_dotenv()

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
# Prepend the directory containing ffmpeg to the PATH.
os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
# Optionally, set an environment variable for ffmpeg.
os.environ["FFMPEG_BINARY"] = ffmpeg_path

# Debug print to confirm
print("ffmpeg_path:", ffmpeg_path, "exists?", os.path.exists(ffmpeg_path))
print("Updated PATH:", os.environ["PATH"])

ssl._create_default_https_context = ssl._create_unverified_context
whisper_model = whisper.load_model("base")
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

def convert_audio_torchaudio(input_path, output_path="converted_audio.wav"):
    """Converts an audio file (e.g. MP3) to WAV format using torchaudio."""
    try:
        waveform, sample_rate = torchaudio.load(input_path)
        # Save as WAV format; torchaudio.save defaults to WAV if format isn't specified.
        torchaudio.save(output_path, waveform, sample_rate, format="wav")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Audio conversion error: {str(e)}. Ensure the uploaded file is a valid audio file.")

def transcribe_audio(audio_path):
    """Transcribe an audio file using OpenAI Whisper."""
    # If the file is MP3, convert it to WAV first
    if audio_path.lower().endswith(".mp3"):
        converted_audio_path = convert_audio_torchaudio(audio_path)
    else:
        converted_audio_path = audio_path
    result = whisper_model.transcribe(converted_audio_path)
    return result["text"]

def analyze_audio_metrics(audio_path):
    """Extracts audio metrics such as duration, speaking rate, number of pauses, and loudness."""
    # If the file is MP3, convert it to WAV before analysis
    if audio_path.lower().endswith(".mp3"):
        audio_path = convert_audio_torchaudio(audio_path)
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
