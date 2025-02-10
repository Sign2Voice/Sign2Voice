import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ✅ Prevent parallel tokenizer processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Paths to model and tokenizer (adjust if necessary)
adapter_model_path = "/Users/beni/projects/capstone_sl_txt_voice/Gloss2Text2Speech/pretrained"
adapter_config_path = "/Users/beni/projects/capstone_sl_txt_voice/Gloss2Text2Speech/pretrained"

base_model_name = "facebook/nllb-200-3.3B"

# ✅ Load the model
print("📢 Loading the Gloss-to-Text model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

# Load the adapter
model.load_adapter(adapter_model_path)
print("✅ Model and adapter successfully loaded!")

# Test the model
# Sample data structure that represents a list of lists of tuples (Alternatively, a sample text can be entered manually)
data = [[('MONTAG', 0), ('AUCH', 1), ('MEHR', 2), ('WOLKE', 3), ('DANN', 4), ('SONNE', 5), ('UBERWIEGEND', 6), ('REGEN', 7), ('UBERWIEGEND', 8), ('GEWITTER', 9)]]

# Extract only the words/glosses from each inner list
words = []
for sublist in data:
    for word, index in sublist:
        words.append(word)

# Output as string
example_text = " ".join(words)
# example_text = "MONDAY ALSO MORE CLOUDS THAN SUN PREDOMINANTLY RAIN PREDOMINANTLY THUNDERSTORM" # manual example
print(example_text)

inputs = tokenizer(example_text, return_tensors="pt")
output = model.generate(inputs.input_ids, max_length=50)

# Decode and print the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

