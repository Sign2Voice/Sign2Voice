import os
import pyaudio  # For audio playback
from openai import AzureOpenAI  # Azure TTS API
from dotenv import load_dotenv  # To load environment variables from .env file

# Load environment variables from the .env file
load_dotenv()

# Azure OpenAI API credentials from environment variables
AZUREENDPOINT = os.getenv("AZUREENDPOINT")
APIKEY = os.getenv("APIKEY")
AZUREDEPLOYMENT = os.getenv("AZUREDEPLOYMENT")
APIVERSION = os.getenv("APIVERSION")

# Check if API keys are loaded
if not all([AZUREENDPOINT, APIKEY, AZUREDEPLOYMENT, APIVERSION]):
    raise ValueError("❌ Missing Azure OpenAI API credentials. Please check your .env file!")

# Initialize the OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZUREENDPOINT,
    api_key=APIKEY,
    azure_deployment=AZUREDEPLOYMENT,
    api_version=APIVERSION
)

# 🔹 Example text for speech synthesis
example_text = "Am Mittwoch wechselnd bis stark bewölkt, dabei im Norden teils Regen, sonst einige Schauer, in Hochlagen teils mit Schnee vermischt. Im Süden länger trocken mit sonnigen Abschnitten. Höchsttemperaturen 6 bis 12 Grad, mit den höchsten Werten entlang des Oberrheins. Mäßiger, im Norden und im Bergland teils frischer und in Böen starker bis stürmischer Südwest- bis Westwind"

print(f"🔊 Starting TTS with text: {example_text}")

# 🔹 Prepare PyAudio for audio playback
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,  # 16-bit PCM
    channels=1,  # Mono
    rate=24000,  # 24 kHz sample rate
    output=True
)

# 🔹 Function for real-time streaming of speech synthesis
def stream_audio(text):
    with client.audio.speech.with_streaming_response.create(
        model="tts-1-hd",  # High-quality speech model
        voice="nova",  # Choose a voice (e.g., "nova", "alloy", "echo")
        input=text,
        response_format="pcm"
    ) as response:
        for chunk in response.iter_bytes(1024):  # Stream 1024 bytes at a time
            stream.write(chunk)  # Play audio in real-time

# 🔹 Start speech synthesis
stream_audio(example_text)

# 🔹 Close the PyAudio stream
stream.stop_stream()
stream.close()
p.terminate()

print("✅ TTS test completed!")
