import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import warnings and suppress UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path

# Package Import
import pyaudio  # For handling audio playback and recording
from openai import AzureOpenAI  # For Azure OpenAI API
from dotenv import load_dotenv
import os
import streamlit as st
import sys

# Definiere das Gerät als CPU
# device = torch.device('cpu')

# Paths to the files (please customise)
adapter_model_path = "/Users/beni/projects/capstone_sl_txt_voice/Gloss2Text2Speech/pretrained"
adapter_config_path = "/Users/beni/projects/capstone_sl_txt_voice/Gloss2Text2Speech/pretrained"

# Load the base model and the tokeniser
base_model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

# Load the adapter configuration and the model
model.load_adapter(adapter_model_path)

# Verschiebe das Modell auf die CPU
#model.to(device)

# Test the model
# Sample data structure that represents a list of lists of tuples (Alternatively, a sample text can be entered manually)
data = [[('ICH', 0), ('LUFT', 1), ('WETTER', 2), ('GERADE', 3), ('loc-SUEDWEST', 4), ('TEMPERATUR', 5), ('__PU__', 6), ('KUEHL', 7), ('SUED', 8), ('WARM', 9), ('ICH', 10), ('IX', 11)]]

# Extract only the words/glosses from each inner list
words = []
for sublist in data:
    for word, index in sublist:
        words.append(word)

# Output as string
example_text = " ".join(words)
# example_text = "ICH LUFT WETTER GERADE loc-SUEDWEST TEMPERATUR __PU__ KUEHL SUED WARM ICH IX" # manual example
print(example_text)

inputs = tokenizer(example_text, return_tensors="pt")
output = model.generate(inputs.input_ids, max_length=50)

# Decode and print the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)



# Load environment variables from .env file
load_dotenv()

# Access the variables
AZUREENDPOINT=os.getenv("AZUREENDPOINT")
APIKEY=os.getenv('APIKEY')
AZUREDEPLOYMENT=os.getenv('AZUREDEPLOYMENT')
APIVERSION=os.getenv('APIVERSION')

# Define client
client = AzureOpenAI(
    azure_endpoint=AZUREENDPOINT,
    api_key=APIKEY,
    azure_deployment=AZUREDEPLOYMENT,
    api_version=APIVERSION
)

# # Define text
# text = """Am Mittwoch wechselnd bis stark bewölkt, dabei im Norden teils Regen, sonst einige Schauer, 
# in Hochlagen teils mit Schnee vermischt. Im Süden länger trocken mit sonnigen Abschnitten. Höchsttemperaturen 6 bis 12 Grad, 
# mit den höchsten Werten entlang des Oberrheins. Mäßiger, im Norden und im Bergland teils frischer und in Böen starker bis 
# stürmischer Südwest- bis Westwind. In der Nacht zum Donnerstag im Norden, später auch ganz im Westen etwas Regen, 
# im Süden und Osten teils gering bewölkt und trocken. Später Nebelbildung. Tiefstwerte im Süden und im Bergland 0 bis -5 Grad, 
# sonst 4 bis 0 Grad."""

# Streamlit-Layout
st.title("Echtzeit Text-to-Speech")
# st.write("Geben Sie einen Text ein. Die Sprachausgabe wird nach der Eingabe an das Open AI TTS model űbergeben und direkt gestreamt.")
# Text-Eingabefeld
# text = st.text_area("Text eingeben", height=200)

text = output_text

if text.strip(): #if text is entered into text area

    #Script for not saving file, but streaming it in realtime before the whole text was transformed via the tts model
    #HD Model

    # Initialize PyAudio, which provides bindings for PortAudio, a cross-platform audio library
    p = pyaudio.PyAudio()

    # Open a stream with specific audio format parameters
    stream = p.open(format=pyaudio.paInt16,  # Format: 16-bit PCM (Pulse Code Modulation)
                    channels=1,              # Channels: 1 (Mono)
                    rate=24000,              # Sample rate: 24,000 Hz (samples per second)
                    output=True)             # Stream opened for output (playback)

    # Function to stream and play audio in real-time
    def stream_audio():
        # Create a TTS (Text-to-Speech) request
        with client.audio.speech.with_streaming_response.create(
            model="tts-1-hd", # Specify the TTS model to use
            voice="nova", # Specify the voice to use for TTS
            input=text,  # Input text to be converted to speech
            response_format="pcm" # Response format: PCM (Pulse Code Modulation)
        ) as response:
            # Iterate over the audio chunks in the response
            for chunk in response.iter_bytes(1024):  # Read 1024 bytes at a time
                stream.write(chunk)  # Write each chunk to the PyAudio stream for playback

    # Start streaming and playing the audio
    stream_audio()

    # Close the PyAudio stream properly
    stream.stop_stream()  # Stop the stream
    stream.close()        # Close the stream
    p.terminate()         # Terminate the PyAudio session

# Exit the script explicitly
sys.exit()