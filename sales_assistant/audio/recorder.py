import os
import pyaudio
import wave
import speech_recognition as sr
import threading

class AudioRecorder:
    def __init__(self, audio_file="audio/conversation.wav", record_seconds=10):
        self.audio_file = audio_file
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.frames = []
        self.is_recording = False
        self.record_seconds = record_seconds
        
        # Ensure the audio directory exists
        audio_dir = os.path.dirname(audio_file)
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

    def record_audio(self):
        """
        Records audio until Enter is pressed again and saves it to a file.
        """
        print("Press Enter to start recording...")
        input()
        self.frames = []
        self.is_recording = True
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)

        print("Recording... Press Enter to stop.")
        
        def listen_for_audio():
            while self.is_recording:
                data = stream.read(self.chunk)
                self.frames.append(data)
        
        # Start listening for audio in a separate thread
        recording_thread = threading.Thread(target=listen_for_audio)
        recording_thread.start()

        # Wait for user input to stop the recording
        input("Press Enter to stop recording...\n")
        self.is_recording = False
        recording_thread.join()

        # Stop the audio stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recorded audio to a file
        with wave.open(self.audio_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
        
        print("Recording stopped and saved to file.")
        return self.audio_file

    def transcribe_audio(self):
        """
        Converts the recorded audio file to text using Google Speech Recognition.
        """
        recognizer = sr.Recognizer()
        with sr.AudioFile(self.audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                print(f"Transcribed Text: {text}")
                return text
            except sr.UnknownValueError:
                print("Could not understand the audio.")
                return ""
            except sr.RequestError as e:
                print(f"Speech Recognition error: {e}")
                return ""

    def listen_and_transcribe(self):
        """
        Records audio and returns the transcript.
        """
        self.record_audio()
        return self.transcribe_audio()
