#!python3

from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Initialize the pipeline and specify the device
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, device="mps")

# Generate text
result = pipe("Once upon a time,", max_length=50, num_return_sequences=1)

# Extract and print the generated text
generated_text = result[0]['generated_text']
print(generated_text)
