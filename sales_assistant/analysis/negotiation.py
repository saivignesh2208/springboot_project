from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class NegotiationAssistant:
    def __init__(self, model_name="google/flan-t5-xl"):
        """
        Initialize the model and tokenizer for generating negotiation strategies.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_negotiation_suggestions(self, transcript, sentiment, intent):
        """
        Generates negotiation suggestions (not just one-line responses) to help close the deal.
        """
        # Create a tailored prompt for sales strategies
        prompt = f"""
        You are a highly skilled sales assistant. Your goal is to help a salesperson close a deal.
        Based on the following information, suggest up to three actionable strategies to convince the buyer:
        
        Transcript: "{transcript}"
        Buyer sentiment: "{sentiment}"
        Buyer intent: "{intent}"
        
        Examples of strategies include:
        - Explaining the benefits of the product in more detail.
        - Offering a discount or promotion.
        - Addressing the buyer's specific concerns.
        - Changing the topic to a positive aspect of the product.
        - Suggesting a follow-up meeting or call.
        
        Please provide clear and concise suggestions.
        """

        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        # Generate output
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=150,  # Allow space for multiple suggestions
            num_beams=5,    # Use beam search for better quality
            temperature=0.7,  # Balance between creativity and coherence
            early_stopping=True
        )

        # Decode and return the suggestions
        suggestions = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return suggestions
