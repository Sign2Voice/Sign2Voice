# Sign2Voice 🗣️
We want to develop an application that enables deaf people to communicate with hearing individuals who do not know sign language. The app should recognize signs, translate them into text, and then convert this text into spoken language.

## Features
- Recognition of sign language gestures from individual images (frames)
- Translation of gestures into glosses using a neural network
- Conversion of glosses into natural text
- Speech output of the generated text (Text-to-Speech)

## Architecture of Sign2Voice
Our system is structured in a modular pipeline, where each component plays a specific role in processing sign language gestures and converting them into speech. The architecture consists of four main stages:

### 1️⃣ Frame Processing (Input)
- The application takes image frames as input.
- These frames are either uploaded manually or extracted from a video.
- OpenCV is used to preprocess the images before sending them to the recognition model.

### 2️⃣ Sign Language Recognition (Gloss Prediction)
- We use **CorrNet**, a deep learning model, to recognize glosses (simplified representations of sign language).
- CorrNet processes the input frames and predicts a sequence of glosses, which represent the recognized signs.
- The output at this stage is a structured sequence of glosses that will later be converted into natural text.

### 3️⃣ Gloss-to-Text Translation
- The predicted glosses are translated into full sentences using our **Gloss2Text** model.
- Gloss2Text ensures that the generated sentences are grammatically correct and contextually meaningful.
- The final output of this step is a fully structured natural language sentence, making it easier for non-signing people to understand

### 4️⃣ Text-to-Speech Conversion (Output)
- The generated sentence is sent to the OpenAI Text-to-Speech Model via the Audio API.
- The model transforms the text into spoken words using OpenAI's advanced speech synthesis technology.
- This allows for seamless playback in the Streamlit application, enhancing the user experience.

### How the Components Work Together
- 1️⃣ User uploads frames (or extracts them from a video).
- 2️⃣ CorrNet processes the frames and predicts glosses.
- 3️⃣ Gloss2Text translates the glosses into full sentences.
- 4️⃣ The final text is converted into real-time speech using OpenAI's Text-to-Speech API and streamed via PyAudio.

## Requirements:
- pyenv with Python: 3.9.17

### Setup
Install the virtual environment and the required packages by following commands:

```BASH
pyenv local 3.9.17
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev.txt
```

The `requirements.txt` file contains the libraries needed for deployment.

### Preperation for CorrNet
Detailed instructions can be found in the CorrNet folder readme.
CorrNet requires a pretrained model to perform sign language recognition.
1. Create a directory for the model: `CorrNet/pretrained_model`
2. [Download](https://drive.google.com/file/d/1Xt_4N-HjEGlVyrMENydsxNtpVMSg5zDb/view) the pretrained CorrNet model and place it in the diretory

### Preperation for Gloss2Text
Please [download](https://drive.google.com/file/d/1eoV_DNfuEXXSLMCM3WwHgPzgGEuWLCSD/view?usp=sharing) the file adapter_model.bin (100 MB / final adapter model) and put it into the `Gloss2Text2Speech/pretrained` folder.

### Preperation for TextToSpeech
To run the audio file, please create an .env file and put it into the main directory
```BASH
AZUREENDPOINT=
APIKEY=
AZUREDEPLOYMENT=
APIVERSION=
```
If you want to test the model and you need the .env please write us an inquiry.


## Usage
### Frames to audio output
To run the application, use the following command:
```streamlit run st_to_txt/streamlit_app.py```
### One test video to glosses
To run the application, use the following command:
```streamlit run st_to_txt/streamlit_video_app.py```

## Future Improvements
Enhancing the system for better performance and usability:
- ✅ Optimize processing speed – Improve model efficiency to reduce processing time
- ✅ Enable real-time video input – Allow users to process continuous video streams instead of individual frames

## Resources & References
Our project is built upon several open-source frameworks and pretrained models. Here are the original repositories and resources we used:

### CorrNet (Sign Language Recognition Model)
- Repository: [CorrNet](https://github.com/hulianyuyy/CorrNet)
- Paper: [CorrNet Research Paper](https://arxiv.org/abs/2303.03202)

### Gloss2Text (Text Generation from Glosses)
- Repository: Gloss2Text Model
- Paper: Gloss2Text Paper
 





  


