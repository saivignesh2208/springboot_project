from audio.recorder import AudioRecorder
from analysis.sentiment_analysis import SentimentIntentClassifier
from analysis.negotiation import NegotiationAssistant

def main():
    recorder = AudioRecorder()
    print("Press Enter to start listening and generating transcript (Ctrl+C to exit)...")
    
    # Capture transcript from audio
    transcript = recorder.listen_and_transcribe()

    if transcript:
        print(f"Generated Transcript:\n{transcript}")
        
        # Initialize the sentiment and intent classifier
        classifier = SentimentIntentClassifier()

        try:
            # Classify sentiment and intent
            sentiment, intent = classifier.classify(transcript)
            print(f"Sentiment: {sentiment}")
            print(f"Intent: {intent}")
        except ValueError as e:
            print(f"Error in sentiment/intent classification: {e}")
            return

        # Initialize the negotiation assistant
        assistant = NegotiationAssistant()

        # Generate actionable negotiation suggestions
        suggestions = assistant.generate_negotiation_suggestions(transcript, sentiment, intent)
        print("Negotiation Suggestions:")
        print(suggestions)

    else:
        print("No transcript generated.")

if __name__ == "__main__":
    main()
