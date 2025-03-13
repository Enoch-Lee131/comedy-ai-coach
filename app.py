import streamlit as st
from comedy_ai import joke_feedback, transcribe_audio, analyze_audio_metrics
import matplotlib.pyplot as plt

st.title("🎤 AI Comedy Coach")

option = st.radio("Choose your input type:", ["Text", "Audio"])

if option == "Text":
    joke_text = st.text_area("Enter your joke:", height=150)
    if st.button("Analyze My Joke"):
        if joke_text.strip():
            with st.spinner("Analyzing your joke..."):
                feedback = joke_feedback(joke_text)
                st.markdown("## 📝 Feedback:")
                st.write(feedback)
        else:
            st.warning("Please enter a joke to analyze.")

elif option == "Audio":
    audio_file = st.file_uploader("Upload your audio file (MP3 or WAV)", type=["mp3", "wav"])

    if audio_file is not None:
        # Use the original file extension for saving
        file_extension = audio_file.name.split('.')[-1]
        temp_audio_path = f"temp_audio.{file_extension}"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        with st.spinner("Transcribing your audio..."):
            transcript = transcribe_audio(temp_audio_path)
            st.markdown("### 🗣 Transcription:")
            st.write(transcript)

        with st.spinner("Analyzing your joke delivery..."):
            feedback = joke_feedback(transcript)
            st.markdown("## 📝 Joke Feedback:")
            st.write(feedback)

            # Analyze audio metrics and plot them
            audio_metrics = analyze_audio_metrics(temp_audio_path)
            st.markdown("## 🎙 Delivery Analysis:")
            st.write(f"- Duration: {audio_metrics['duration_seconds']:.2f} seconds")
            st.write(f"- Speaking Rate: {audio_metrics['speaking_rate']:.2f} samples/sec")
            st.write(f"- Number of Pauses: {audio_metrics['num_pauses']}")
            st.write(f"- Average Loudness: {audio_metrics['avg_loudness']:.4f}")

            st.markdown("### 🎙️ Delivery Metrics:")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(
                ["Speaking Rate", "Pauses", "Avg Loudness"],
                [audio_metrics["speaking_rate"], audio_metrics["num_pauses"], audio_metrics["avg_loudness"]],
                color=['skyblue', 'orange', 'green']
            )
            ax.set_xlabel("Metrics")
            st.pyplot(fig)
