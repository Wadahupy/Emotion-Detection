import pyaudio
import librosa
import numpy as np
import time
import streamlit as st
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
import altair as alt
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

# Preprocess audio for prediction
def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return np.expand_dims(mfccs, axis=(0, -1))
    except Exception as e:
        return str(e)

# Streamlit layout
st.title("Assessing Emotional State in EARIST Students through Machine Learning Analysis of Vocal Tone During Pocket Book Reading")
st.write("This App Predicts Emotions Based on Text and Audio Input")

# Text-based emotion prediction
def predict_text_emotion(text):
    predicted_probabilities = text_model.predict_proba([text])[0]
    confidence_text = {emotion: float(prob) for emotion, prob in zip(text_labels, predicted_probabilities)}
    predicted_emotion_text = text_labels[np.argmax(predicted_probabilities)]
    return predicted_emotion_text, confidence_text

# Audio-based emotion prediction
def predict_audio_emotion(file_path):
    features = preprocess_audio(file_path)
    if isinstance(features, str):
        return None, {"error": features}

    prediction = audio_model.predict(features)
    predicted_class = np.argmax(prediction)
    confidence_audio = {label: float(prediction[0][i]) for i, label in enumerate(audio_labels)}
    predicted_emotion_audio = audio_labels[predicted_class]

    return predicted_emotion_audio, confidence_audio

# Utility function to clean up temporary files
def cleanup_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Initialize session state variables
for key in ['confidence_df_text', 'confidence_df_audio', 'predicted_emotion_text', 'predicted_emotion_audio']:
    st.session_state.setdefault(key, None)

# Text input and prediction
text_input = st.text_area("Enter text to predict emotion:", placeholder="Enter your text script here", max_chars=300)
if st.button("Predict Text Emotion"):
    if not text_input.strip():
        st.error("Please enter text before clicking the Predict button.")
    else:
        predicted_emotion_text, confidence = predict_text_emotion(text_input)
        st.session_state['predicted_emotion_text'] = predicted_emotion_text
        st.session_state['confidence_df_text'] = pd.DataFrame(confidence.items(), columns=["Emotion", "Confidence"])
# Displaying text prediction if already available in session state
if st.session_state.get('predicted_emotion_text'):
    st.success(f"Predicted Text Emotion: {st.session_state['predicted_emotion_text'].title()}")
if st.session_state.get('confidence_df_text') is not None:
    st.write("Text Emotion Confidence")
    st.bar_chart(st.session_state['confidence_df_text'].set_index('Emotion')['Confidence'])

choice = option_menu(
    menu_title=None,
    options=["Upload", "Realtime"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "5px", "background-color": "#262730"},
        "nav-link": {"font-size": "16px", "margin": "0px", "padding": "5px 20px", "border-radius": "5px"},
        "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
    }
)

def live_prediction():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, frames_per_buffer=1024, input=True)
    st.write("Listening...")

    bar_chart_placeholder = st.empty()  # Placeholder for updating the bar chart
    waveform_placeholder = st.empty()  # Placeholder for updating the audio waveform
    emotion_text_placeholder = st.empty()  # Placeholder for displaying emotion text
    confidence_placeholder = st.empty()  # Placeholder for dynamically updating confidence levels

    try:
        while True:
            data = stream.read(1024)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            # Plot the live audio waveform
            fig, ax = plt.subplots()
            ax.plot(audio_data)
            ax.set_title("Live Audio Waveform")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            waveform_placeholder.pyplot(fig)
            
            mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40, n_fft=1024).T, axis=0)
            mfccs = np.expand_dims(np.expand_dims(mfccs, axis=-1), axis=0)

            # Predict the emotion
            prediction = audio_model.predict(mfccs)
            predicted_class = np.argmax(prediction)
            predicted_label = audio_labels[predicted_class]

            # Prepare confidence data
            confidence_data = {label: confidence for label, confidence in zip(audio_labels, prediction[0])}
            confidence_df = pd.DataFrame(list(confidence_data.items()), columns=["Emotion", "Confidence"])

            # Display predicted emotion and confidence levels
            emotion_text_placeholder.text(f"Predicted Emotion: {predicted_label}")
            confidence_placeholder.text(f"Confidence Levels:\n" + "\n".join(f"{label}: {confidence:.2f}" for label, confidence in confidence_data.items()))
            
            bar_chart_placeholder.bar_chart(confidence_df.set_index("Emotion")["Confidence"])
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

if choice == "Upload":
    st.subheader("File Audio Input")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if audio_file:
        temp_file_path = f"temp_{audio_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(audio_file.read())
        predicted_emotion_audio, audio_confidence = predict_audio_emotion(temp_file_path)
        if predicted_emotion_audio:
            st.session_state['predicted_emotion_audio'] = predicted_emotion_audio
            st.session_state['confidence_df_audio'] = pd.DataFrame(audio_confidence.items(), columns=["Emotion", "Confidence"])
        cleanup_file(temp_file_path)
        
    # Displaying text prediction if already available in session state
    if st.session_state.get('predicted_emotion_audio'):
        st.success(f"Predicted Text Emotion: {st.session_state['predicted_emotion_audio'].title()}")
    if st.session_state.get('confidence_df_audio') is not None:
        st.write("Audio Emotion Confidence")
        st.bar_chart(st.session_state['confidence_df_audio'].set_index('Emotion')['Confidence'])

elif choice == "Realtime":
    st.subheader("Realtime Audio Emotion")
    live_prediction()

st.subheader("RESULT SUMMARY")
col1, col2 = st.columns(2)
if st.session_state.get('confidence_df_text') is not None:
    with col1:
        st.text(f"Predicted Text Emotion: {st.session_state['predicted_emotion_text'].title()}")
        # Create pie chart for text emotion confidence using Altair
        text_chart = alt.Chart(st.session_state['confidence_df_text']).mark_arc().encode(
            theta=alt.Theta(field="Confidence", type="quantitative"),
            color=alt.Color(field="Emotion", type="nominal"),
            tooltip=["Emotion", "Confidence"]
        ).properties(title="Text Emotion Confidence")
        st.altair_chart(text_chart, use_container_width=True)

if st.session_state.get('confidence_df_audio') is not None:
    with col2:
        st.text(f"Predicted Audio Emotion: {st.session_state['predicted_emotion_audio'].title()}")
        # Create pie chart for audio emotion confidence using Altair
        audio_chart = alt.Chart(st.session_state['confidence_df_audio']).mark_arc().encode(
            theta=alt.Theta(field="Confidence", type="quantitative"),
            color=alt.Color(field="Emotion", type="nominal"),
            tooltip=["Emotion", "Confidence"]
        ).properties(title="Audio Emotion Confidence")
        st.altair_chart(audio_chart, use_container_width=True)
