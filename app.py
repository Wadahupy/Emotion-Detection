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
from pydub import AudioSegment

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

# Function to handle MP3 to WAV conversion
def convert_mp3_to_wav(mp3_file_path):
    wav_file_path = mp3_file_path.replace(".mp3", ".wav")
    try:
        audio = AudioSegment.from_file(mp3_file_path)
        audio.export(wav_file_path, format="wav")
        os.remove(mp3_file_path)
    except Exception as e:
        return None, str(e)
    return wav_file_path, None

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

def live_prediction(line_index, expected_emotion):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, frames_per_buffer=1024, input=True)
    st.write(f"Listening for Line {line_index + 1}...")

    bar_chart_placeholder = st.empty()
    emotion_text_placeholder = st.empty()
    match_placeholder = st.empty()
    table_placeholder = st.empty()

    # session state stores results per line
    if "line_predictions" not in st.session_state:
        st.session_state["line_predictions"] = {}

    try:
        for _ in range(10):  
            data = stream.read(1024)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40, n_fft=2048, hop_length=512)


            mfccs = np.mean(mfccs.T, axis=0)

            mfccs = np.expand_dims(np.expand_dims(mfccs, axis=-1), axis=0)

            prediction = audio_model.predict(mfccs)
            predicted_class = np.argmax(prediction)
            predicted_label = audio_labels[predicted_class]

            # Store confidence data
            confidence_data = {label: confidence for label, confidence in zip(audio_labels, prediction[0])}
            confidence_df = pd.DataFrame(list(confidence_data.items()), columns=["Emotion", "Confidence"])

            # ✅ Store results in session state for the specific line
            st.session_state["line_predictions"][line_index] = {
                "predicted_emotion": predicted_label,
                "confidence_df": confidence_df
            }

            # Update UI
            emotion_text_placeholder.markdown(
                f"Audio Emotion (Line {line_index + 1}): <span style='color:yellow;'>🎤 {predicted_label.capitalize()}</span>", 
                unsafe_allow_html=True
            )
            table_placeholder.dataframe(confidence_df)
            bar_chart_placeholder.bar_chart(confidence_df.set_index("Emotion")["Confidence"])

            if predicted_label == expected_emotion:
                match_placeholder.success('SCRIPT AND AUDIO EMOTION MATCH', icon="✅")
                st.toast('SCRIPT AND AUDIO EMOTION MATCH', icon="✅")
            else:
                match_placeholder.error('SCRIPT AND AUDIO EMOTION DO NOT MATCH!', icon="❌")
                st.toast('SCRIPT AND AUDIO EMOTION MATCH', icon="❌")

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
        with st.container():
            st.markdown(f"#### 🎭 Line {index+1}: {speaker}")
            st.markdown(f"📜 **Text:** {text}")
            st.markdown(f"📝 **Predicted Text Emotion:** <span style='color:yellow;'>{text_emotion.capitalize()}</span>", unsafe_allow_html=True)

            # Display emotion confidence table for text
            text_confidence_df = pd.DataFrame(list(zip(text_labels, text_probabilities)), columns=["Emotion", "Confidence"])
            st.dataframe(text_confidence_df, hide_index=True, width=400)

            # Popover for Audio Upload & Recording
            with st.expander("🎤 **Upload or Record Audio**"):
                col1, col2 = st.columns([1, 1])

                with col1:
                    audio_file = st.file_uploader(f"Upload Audio for Line {index+1}", type=['wav', 'mp3'], key=f"audio_upload_{index}")

                with col2:
                # Center the Record Button Vertically
                    st.markdown(
                        """
                        <style>
                            .center-button {
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                padding-top: 50px;
                                height: 100%;
                            }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.container():
                        st.markdown('<div class="center-button">', unsafe_allow_html=True)
                        record_btn = st.button(f"🎙️ Record Audio", key=f"record_btn_{index}")
                        st.markdown('</div>', unsafe_allow_html=True)

                if record_btn:
                    expected_emotion = text_emotion
                    live_prediction(index, text_emotion)

                if audio_file:
                    temp_file = f"temp_{audio_file.name}"
                    with open(temp_file, "wb") as f:
                        f.write(audio_file.read())

                    # Handle MP3 conversion if necessary
                    if temp_file.endswith('.mp3'):
                        temp_file, error = convert_mp3_to_wav(temp_file)
                        if error:
                            st.error(f"Error converting MP3 to WAV: {error}")
                            continue

                    # Predict emotion from audio
                    audio_emotion, confidence, probabilities = predict_audio_emotion(temp_file)

                    # Show emotion confidence for audio
                    audio_confidence_df = pd.DataFrame(list(zip(audio_labels, probabilities)), columns=["Emotion", "Confidence"])
                    st.write("🎵 **Emotion Confidence (Audio):**")
                    st.dataframe(audio_confidence_df, hide_index=True, width=400)

                    # Match check
                    if audio_emotion == text_emotion:
                        st.markdown(f"🎤 **Predicted Audio Emotion:** <span style='color:yellow;'>{audio_emotion.capitalize()}</span>", unsafe_allow_html=True)
                        st.success('SCRIPT AND AUDIO EMOTION MATCH', icon='✅')
                        st.toast('SCRIPT AND AUDIO EMOTION MATCH', icon='✅')
                    else:
                        st.markdown(f"🎤 **Audio Emotion:** <span style='color:yellow;'>{audio_emotion.capitalize()}</span>", unsafe_allow_html=True)
                        st.error('SCRIPT AND AUDIO EMOTION DO NOT MATCH', icon='❌')
                        st.toast('SCRIPT AND AUDIO EMOTION DO NOT MATCH', icon='❌')

                    os.remove(temp_file)
