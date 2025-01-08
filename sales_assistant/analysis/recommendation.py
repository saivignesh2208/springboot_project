from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_flan_t5(prompt):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate output
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=200,  # Set the maximum length for the generated output
        num_beams=5,  # Number of beams for beam search
        temperature=0.7,  # Adjust creativity of the output
        early_stopping=True
    )
    
    # Decode and return the output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example prompt
prompt = "You are a sales assistant. Provide a convincing one-line response to sell a product to a customer who shows interest."

# Test the model
response = test_flan_t5(prompt)
print("Generated Response:", response)
