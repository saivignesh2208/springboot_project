from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SentimentIntentClassifier:
    def __init__(self, model_name="google/flan-t5-xl"):
        """
        Initialize the model and tokenizer for classification.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def classify(self, transcript):
        """
        Classify sentiment and intent from the transcript using the LLM.
        """
        prompt = f"""
        You are an expert in analyzing buyer conversations. Based on the transcript below, classify the sentiment and intent:
        
        Transcript: "{transcript}"
        
        Provide the sentiment (e.g., POSITIVE, NEGATIVE, NEUTRAL) and intent (e.g., interest, hesitation, rejection) in this format: sentiment; intent.
        """

        # Tokenize and process the input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        # Generate output
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=50,  # Limit the length of the classification
            num_beams=5,
            temperature=0.7,
            early_stopping=True
        )

        # Decode the result
        classification = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Debugging: Print the raw output
        print(f"Raw Model Output: {classification}")

        # Attempt to parse the output
        try:
            sentiment, intent = classification.split(";")  # Expecting "sentiment; intent"
            return sentiment.strip(), intent.strip()
        except ValueError:
            # Handle unexpected format
            print("Unexpected model output format. Defaulting to NEUTRAL sentiment and unknown intent.")
            return "NEUTRAL", "unknown"
