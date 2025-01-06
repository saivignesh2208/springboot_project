import pyaudio
import wave
import os
import speech_recognition as sr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import json

# File paths
AUDIO_FILE = "conversation.wav"
TEXT_FILE = "conversation_log.txt"
SUMMARY_FILE = "summary_log.txt"
EXCEL_FILE = "salesman_data.xlsx"  # Excel file for product data

# Recording Parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
STOP_NEGOTIATION = False

# Load Hugging Face pipelines for sentiment analysis and summarization
sentiment_analyzer = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load FLAN-T5 model for negotiation suggestion
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")

# Read product database from Excel
try:
    product_database = pd.read_excel(EXCEL_FILE).to_dict(orient='records')
    if not product_database:
        raise ValueError("Product database is empty.")
except Exception as e:
    print(f"Error loading product database: {e}")
    product_database = []


# Function to record audio
def record_audio():
    print("Press Enter to start recording...")
    input()
    global STOP_NEGOTIATION
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print("Recording... Press Enter to stop recording.")
    try:
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("Recording stopped.")
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    # Save audio file
    wf = wave.open(AUDIO_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


# Function to convert audio to text
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"Transcribed Text: {text}")
            with open(TEXT_FILE, "a") as file:
                file.write(text + "\n")
            return text
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"Speech Recognition error: {e}")
            return ""


# Function for sentiment analysis
def analyze_sentiment_and_intent(text):
    sentiment = sentiment_analyzer(text)[0]
    print(f"Sentiment Analysis: {sentiment}")
    return sentiment


# Function to generate negotiation suggestion
def generate_negotiation_suggestion(text, sentiment, database):
    try:
        # Debugging: Check database structure
        print("Product database preview:", database[:3])

        # Summarize product database (handle missing keys gracefully)
        database_summary = [
            f"{item.get('Name', 'Unknown')} (${item.get('Price', 'N/A')})"
            for item in database[:3]
        ]

        # Construct prompt based on the conversation and sentiment analysis
        conversation_context = f"Customer sentiment: {sentiment['label']} ({sentiment['score']:.2f})"
        input_text = (
            f"Conversation transcript: {text}\n"
            f"{conversation_context}\n"
            f"Product details: {', '.join(database_summary)}\n"
            f"Generate a negotiation suggestion tailored to the sentiment and context:"
        )

        # Tokenize and generate output from FLAN-T5 model
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        outputs = model.generate(
            inputs['input_ids'],
            max_length=150,
            num_beams=5,
            early_stopping=True,
            temperature=0.7,
            top_p=0.9
        )

        suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nNegotiation Suggestion:\n{suggestion}")

        # Save the negotiation suggestion
        with open(SUMMARY_FILE, "a") as file:
            file.write(suggestion + "\n")
    except KeyError as e:
        print(f"Error: Missing key in database: {e}")
    except Exception as e:
        print(f"Error in generating negotiation suggestion: {e}")


# Main logic
def main():
    global STOP_NEGOTIATION
    try:
        while not STOP_NEGOTIATION:
            record_audio()  # Start recording
            text = audio_to_text(AUDIO_FILE)  # Convert audio to text
            if text:
                print("Press Enter to analyze and negotiate or Ctrl+C to exit.")
                input()
                sentiment = analyze_sentiment_and_intent(text)
                generate_negotiation_suggestion(text, sentiment, product_database)  # Perform negotiation logic
    except KeyboardInterrupt:
        STOP_NEGOTIATION = True
        print("\nTerminating program...")


if __name__ == "__main__":
    main()

