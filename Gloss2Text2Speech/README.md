<div align="center">

# GLOSS2TEXT & Text2Speech


## 1. GLOSS2TEXT: Sign Language Gloss translation using LLMs and Semantically Aware Label Smoothing

[![arXiv](https://img.shields.io/badge/arXiv-GLOSS2TEXT-A10717.svg?logo=arXiv)](https://arxiv.org/abs/2407.01394)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</div>

## Description
Official PyTorch implementation of the paper:
<div align="center">

[GLOSS2TEXT: Sign Language Gloss translation using LLMs and Semantically Aware Label Smoothing](https://aclanthology.org/2024.findings-emnlp.947/).

</div>

### Pre-trained model
Use of a pre-trained model with the support of the author Pooya Fayyazsanavi. Thanks to:

```
@inproceedings{fayyazsanavi-etal-2024-gloss2text,
    title = "{G}loss2{T}ext: Sign Language Gloss translation using {LLM}s and Semantically Aware Label Smoothing",
    author = "Fayyazsanavi, Pooya  and
      Anastasopoulos, Antonios  and
      Kosecka, Jana",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.947",
    pages = "16162--16171",
    abstract = "Sign language translation from video to spoken text presents unique challenges owing to the distinct grammar, expression nuances, and high variation of visual appearance across different speakers and contexts. Gloss annotations serve as an intermediary to guide the translation process. In our work, we focus on \textit{Gloss2Text} translation stage and propose several advances by leveraging pre-trained large language models (LLMs), data augmentation, and novel label-smoothing loss function exploiting gloss translation ambiguities improving significantly the performance of state-of-the-art approaches. Through extensive experiments and ablation studies on the PHOENIX Weather 2014T dataset, our approach surpasses state-of-the-art performance in \textit{Gloss2Text} translation, indicating its efficacy in addressing sign language translation and suggesting promising avenues for future research and development.",
}

```
<div align="center">

## 2. Text2Speech
Stream and play audio in real-time. TTS model used: tts-1-hd

</div>

## Installation
To set up the environment, run:

```
pip install -r requirements.txt
```
Python version 3.8.20., requirements in the datafolder *Gloss2Text2Speech*.

Alternatively, a virtual environment *slt* can be created with Anaconda:

```
conda create -n slt python=3.8.20
```

## Dataset
We used a pre-trained model trained on the [Phoenix-2014T dataset](
https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/), a German continuous sign language recognition and translation benchmark data set with gloss and translation pairs.

## Preparation
The following data for the *pre-trained model* was kindly provided by the author
Pooya Fayyazsanavi:

 *Already in the repository in the data folder 'pretrained'*
- adapter_config.json
- README.md

Please [download](https://drive.google.com/file/d/1eoV_DNfuEXXSLMCM3WwHgPzgGEuWLCSD/view?usp=sharing) additionally the file 

- adapter_model.bin

(100 MB / final adapter model) and put it into the `Gloss2Text2Speech/pretrained` folder.

## Infrastructure

To run the audio file, please create an .env file with your:

AZUREENDPOINT=   
APIKEY=  
AZUREDEPLOYMENT=  
APIVERSION=  

## Testing the pre-trained model
Make sure that the paths in the code are updated. 
To start testing, run the following command. Modify any arguments as needed:

```
python model_g2t_t2s_new.py
```
To open it in Streamlit, please enter it directly in the terminal (please adjust path
):

```
streamlit run /Users/...../Gloss2Text2Speech/
model_g2t_t2s_new.py
```
## License:
This code depends on several libraries, including PyTorch, HuggingFace, and Two-Stream Network. It also uses the Phoenix-2014T dataset. Please ensure compliance with their respective licenses.
