import pyaudio
import librosa
import numpy as np
import time
import streamlit as st
import pandas as pd
import joblib
import os
import wave
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load models and labels
@st.cache_resource
def load_text_model():
    return joblib.load("text_emotion.pkl")

@st.cache_resource
def load_audio_model():
    return load_model('lstm_model.keras')

text_model = load_text_model()
audio_model = load_audio_model()

text_labels = text_model.classes_
audio_labels = ['angry', 'disgust', 'joy', 'fear', 'neutral', 'ps', 'sad']

# Initialize session state variables
for key in ['confidence_df_text', 'confidence_df_audio', 'predicted_emotion_text', 'predicted_emotion_audio']:
    st.session_state.setdefault(key, None)

if "line_predictions" not in st.session_state:
    st.session_state["line_predictions"] = {}

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, duration=5, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return np.expand_dims(mfccs, axis=(0, -1))

def predict_text_emotion(text):
    predicted_probabilities = text_model.predict_proba([text])[0]
    predicted_emotion = text_labels[np.argmax(predicted_probabilities)]
    confidence = float(np.max(predicted_probabilities))

    st.session_state.predicted_emotion_text = predicted_emotion
    st.session_state.confidence_df_text = pd.DataFrame(
        list(zip(text_labels, predicted_probabilities)), columns=["Emotion", "Confidence"]
    )
    return predicted_emotion, confidence, predicted_probabilities

def predict_audio_emotion(file_path):
    features = preprocess_audio(file_path)
    prediction = audio_model.predict(features)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    # Storing the results in session state
    st.session_state.predicted_emotion_audio = audio_labels[predicted_class]
    st.session_state.confidence_df_audio = pd.DataFrame(
        list(zip(audio_labels, prediction[0])), columns=["Emotion", "Confidence"]
    )
    return audio_labels[predicted_class], confidence, prediction[0]

def record_audio(filename, duration=5, rate=22050, chunk=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
    st.write("Audio recording")
    frames = []
    for _ in range(int(rate / chunk * duration)):
        frames.append(stream.read(chunk))
    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

def live_prediction(expected_emotion):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, frames_per_buffer=1024, input=True)
    st.write("Listening...")

    bar_chart_placeholder = st.empty()
    emotion_text_placeholder = st.empty()
    match_placeholder = st.empty()
    confidence_placeholder = st.empty()
    table_placeholder = st.empty()  # Placeholder for the confidence table

    try:
        for _ in range(10):  # Process live audio in short bursts
            data = stream.read(1024)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40)
            mfccs = np.mean(mfccs.T, axis=0)  # Correct shape

            mfccs = np.expand_dims(np.expand_dims(mfccs, axis=-1), axis=0)

            prediction = audio_model.predict(mfccs)
            predicted_class = np.argmax(prediction)
            predicted_label = audio_labels[predicted_class]

            # Collect confidence data
            confidence_data = {label: confidence for label, confidence in zip(audio_labels, prediction[0])}
            confidence_df = pd.DataFrame(list(confidence_data.items()), columns=["Emotion", "Confidence"])

            # Store in session state
            st.session_state.predicted_emotion_live = predicted_label
            st.session_state.confidence_df_audio_live = confidence_df

            # Update placeholders dynamically
            emotion_text_placeholder.text(f"Predicted Emotion: {predicted_label}")
            if predicted_label == expected_emotion:
                match_placeholder.success(f"Audio Emotion: {predicted_label.capitalize()} *SCRIPT AND AUDIO EMOTION MATCH*", icon="✅")
            else:
                match_placeholder.error(f"Audio Emotion: {predicted_label.capitalize()} *SCRIPT AND AUDIO EMOTION DO NOT MATCH!*", icon="❌")

            # # Display confidence levels
            # confidence_placeholder.text("Confidence Levels:\n" + "\n".join(f"{label}: {confidence:.2f}" for label, confidence in confidence_data.items()))

            # Update confidence table
            table_placeholder.dataframe(confidence_df)

            # Update bar chart
            bar_chart_placeholder.bar_chart(confidence_df.set_index("Emotion")["Confidence"])

            time.sleep(0.5)
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

def analyze_script(script):
    lines = script.split('\n')
    results = []
    for i, line in enumerate(lines):
        if ':' in line:
            speaker, content = line.split(':', 1)
            text_emotion, text_confidence, text_probabilities = predict_text_emotion(content.strip())
            results.append((i, speaker.strip(), content.strip(), text_emotion, text_confidence, text_probabilities))
    return results

st.title("Assessing Emotional State in EARIST Students through Machine Learning Analysis of Vocal Tone During Pocket Book Reading")
st.write("Analyze emotions from script and audio input")

script_input = st.text_area("Enter your script (one dialogue per line):", placeholder="Speaker: dialogue", height=200)

if script_input:
    st.write("### Script Analysis")
    script_data = analyze_script(script_input)
    for index, speaker, text, text_emotion, text_confidence, text_probabilities in script_data:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        col1.markdown(f"**{speaker}:** {text} (<span style='color:yellow;'>📝 {text_emotion.capitalize()}</span>)", unsafe_allow_html=True)

        
        # Display the emotion confidence for each line of text
        text_confidence_df = pd.DataFrame(list(zip(text_labels, text_probabilities)), columns=["Emotion", "Confidence"])
        col1.write("Emotion Confidence (Text):")
        col1.dataframe(text_confidence_df)
        
        audio_file = col2.file_uploader(f"Upload Audio {index+1}", type=['wav', 'mp3'], key=f"audio_upload_{index}")
        record_btn = col3.button(f"🎤 Record {index+1}", key=f"record_btn_{index}")
        
        if record_btn:
            expected_emotion = text_emotion
            live_prediction(expected_emotion)
    
        
        if audio_file:
            temp_file = f"temp_{audio_file.name}"
            with open(temp_file, "wb") as f:
                f.write(audio_file.read())
            audio_emotion, confidence, probabilities = predict_audio_emotion(temp_file)
            
            # Show emotion summary for the audio
            audio_confidence_df = pd.DataFrame(list(zip(audio_labels, probabilities)), columns=["Emotion", "Confidence"])
            st.write("Emotion Confidence (Audio):")
            st.dataframe(audio_confidence_df)
            
            if audio_emotion == text_emotion:
                st.success(f"Audio Emotion: {audio_emotion.capitalize()} *SCRIPT AND AUDIO EMOTION MATCH*", icon="✅")
            else:
                st.error(f"Audio Emotion: {audio_emotion.capitalize()} *SCRIPT AND AUDIO EMOTION DO NOT MATCH!*", icon="❌")
            os.remove(temp_file)
