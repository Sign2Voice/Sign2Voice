import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ✅ Dynamically detect the project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script's directory
ADAPTER_MODEL_PATH = os.path.join(BASE_DIR, "pretrained")  # Set adapter path dynamically

print(f"📢 Using Adapter Model Path: {ADAPTER_MODEL_PATH}")

# ✅ Prevent parallel tokenizer processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Load the model
base_model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

# ✅ Load the adapter
model.load_adapter(ADAPTER_MODEL_PATH)
print("✅ Model and adapter successfully loaded!")

def gloss_to_text(gloss_sentence, max_length=50):
    """
    Converts a list of glosses into a natural language sentence.

    :param gloss_sentence: String containing glosses (e.g., "MONDAY MORE CLOUD RAIN")
    :param max_length: Maximum length of generated text
    :return: Generated natural sentence
    """

    print("\n📢 **Input Glosses:**", gloss_sentence)

    # **Tokenize & pass to model**
    inputs = tokenizer(gloss_sentence, return_tensors="pt")
    output = model.generate(inputs.input_ids, max_length=max_length)

    # **Decode & return the result**
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("📝 **Generated Text:**", output_text)

    return output_text