import os
import streamlit as st
import whisper
from gtts import gTTS
import io
from openai import OpenAI  # Import OpenAI for AI/ML API calls
import imageio
imageio.plugins.ffmpeg.download()  # Ensure FFmpeg is downloaded and used by Pydub

from pydub import AudioSegment

# Set the base URL and API key for AI/ML API
base_url = "https://api.aimlapi.com/v1"
api_key = "701b35863e6d4a7b81bdcad2e6f3c880"  # Your API key

# Initialize the OpenAI API with the custom base URL and your API key
api = OpenAI(api_key=api_key, base_url=base_url)

# Load the Whisper model for audio transcription
model = whisper.load_model("base")

# Function to make a chat completion call to the AI/ML API
def call_aiml_api(user_prompt, system_prompt="You are a helpful assistant."):
    try:
        completion = api.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",  # Specify the model from AI/ML
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=256,
        )

        # Return the response from the AI model
        return completion.choices[0].message.content.strip()

    except Exception as e:
        raise Exception(f"API request failed with error: {e}")

# Function to process audio and interact with the AI/ML API
def process_audio(file):
    try:
        # Convert the uploaded file to a format suitable for Whisper
        audio = AudioSegment.from_file(file)
        audio.export("input.wav", format="wav")
        audio_file = "input.wav"

        # Load and transcribe audio using Whisper
        result = model.transcribe(audio_file)
        user_prompt = result["text"]

        # Call AI/ML API to get a response
        response_message = call_aiml_api(user_prompt)

        # Convert response message to speech using gTTS
        tts = gTTS(response_message)
        response_audio_io = io.BytesIO()
        tts.write_to_fp(response_audio_io)  # Save the audio to BytesIO object
        response_audio_io.seek(0)

        # Return the response text and the response audio bytes
        return response_message, response_audio_io

    except Exception as e:
        # Handle any errors
        return f"An error occurred: {e}", None

# Streamlit App Layout
st.title("Voice-to-Voice AI Chatbot with AI/ML API")
st.write("Developed by [Adnan Tariq](https://www.linkedin.com/in/adnaantariq/) with ❤️")

# File uploader for audio input
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Process the uploaded audio file
    with st.spinner('Processing...'):
        response_text, response_audio_io = process_audio(uploaded_file)

    # Display the response text
    st.write(f"**Chatbot Response:** {response_text}")

    # Play the response audio
    if response_audio_io:
        st.audio(response_audio_io, format="audio/mp3")
