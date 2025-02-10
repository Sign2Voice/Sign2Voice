import pyaudio
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# Load API keys from .env file
load_dotenv()
AZUREENDPOINT = os.getenv("AZUREENDPOINT")
APIKEY = os.getenv("APIKEY")
AZUREDEPLOYMENT = os.getenv("AZUREDEPLOYMENT")
APIVERSION = os.getenv("APIVERSION")

# Azure OpenAI Client
client = AzureOpenAI(
    azure_endpoint=AZUREENDPOINT,
    api_key=APIKEY,
    azure_deployment=AZUREDEPLOYMENT,
    api_version=APIVERSION,
)

def text_to_speech(text):
    """
    Converts a given text into speech and plays it through the speakers.

    :param text: The text to be spoken
    """

    print("🔊 Starting TTS with text:", text)

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    # Stream audio
    with client.audio.speech.with_streaming_response.create(
        model="tts-1-hd", voice="nova", input=text, response_format="pcm"
    ) as response:
        for chunk in response.iter_bytes(1024):
            stream.write(chunk)

    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("✅ TTS process completed!")
