import sys
import os
import cv2
import torch
import numpy as np
from collections import OrderedDict

# Add CorrNet to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CorrNet")))
# Now import utils from CorrNet
from utils import video_augmentation

from slr_network import SLRModel

# ✅ **Hauptfunktion für Streamlit**
def process_frames(model_path, frames_dir, language="phoenix", max_frames_num=360):
    """
    Lädt extrahierte Frames, verarbeitet sie durch das Modell und gibt die vorhergesagten Glossen zurück.
    """

    # 🔹 **Automatische Gerät-Erkennung (GPU oder CPU)**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📢 Debug: Nutze Gerät {device}")  # Debugging: Zeigt an, ob CPU oder GPU genutzt wird

    # 🔹 Debugging-Informationen
    print(f"📢 Debug: Frames-Ordner: {frames_dir}")
    print(f"📢 Debug: Modell-Pfad: {model_path}")
    print(f"📢 Debug: Sprache: {language}")

    # 🔹 GPU oder CPU und Gloss-Dictionary laden
    dataset = "phoenix2014" if language == "phoenix" else "CSL-Daily"
    dict_path = f'CorrNet/preprocess/{dataset}/gloss_dict.npy'
    gloss_dict = np.load(dict_path, allow_pickle=True).item()

    print(f"📢 Debug: Verwende Dataset {dataset}")
    print(f"📢 Debug: Gloss-Wörterbuch geladen! {len(gloss_dict)} Einträge")

    # 🔹 Frames laden und sortieren
    def load_frames(frames_dir, max_frames_num=360):
        img_list = []
        img_paths = sorted(
            [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
        )
        for img_path in img_paths[:max_frames_num]:  # Begrenzung auf max. Frames
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)
        return img_list

    print(f"📢 Debug: Lade Frames aus {frames_dir}")
    frames = load_frames(frames_dir, max_frames_num)
    print(f"📢 Debug: {len(frames)} Frames geladen!")

    # 🔹 Frames transformieren
    transform = video_augmentation.Compose([
        video_augmentation.CenterCrop(224), # 🟢 1️⃣ Crop in the middle to 224x224
        video_augmentation.Resize(1.0),     # 🟢 2️⃣ Scaling (1.0 means unchanged)
        video_augmentation.ToTensor(),      # 🟢 3️⃣ Conversion to a tensor format
    ])
    vid, _ = transform(frames, None, None)
    vid = vid.float() / 127.5 - 1  # Normalisierung
    vid = vid.unsqueeze(0).to(device)  # Shape anpassen & auf CPU/GPU schieben

    print(f"📢 Debug: Frames transformiert! Shape: {vid.shape}")

    # 🔹 Modell laden
    model = SLRModel(
        num_classes=len(gloss_dict) + 1,
        c2d_type='resnet18',
        conv_type=2,
        use_bn=1,
        gloss_dict=gloss_dict,
        loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0},
    )

    # 📢 **Modellgewichte laden**
    state_dict = torch.load(model_path, map_location=device)['model_state_dict']
    state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)  # **Modell auf GPU/CPU schieben**
    model.eval()

    print(f"📢 Debug: Modell erfolgreich geladen auf {device}!")

    # 🔹 Modell ausführen
    vid_lgt = torch.LongTensor([vid.size(1)]).to(device)  # Länge auf GPU/CPU schieben
    ret_dict = model(vid, vid_lgt, label=None, label_lgt=None)

    # 🔹 Glossen extrahieren
    glosses = ret_dict['recognized_sents']
    gloss_list = [" ".join([gloss[0] for gloss in gloss_seq]) for gloss_seq in glosses]

    print(f"📢 Endgültige Gloss-Vorhersagen: {gloss_list}")

    return gloss_list  # ✅ Rückgabe für Streamlit


