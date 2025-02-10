import os
import pdb
import time
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder

class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        print(f"📜 Inhalt von gloss_dict: {gloss_dict}")
        

        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}


        # 🛠 Debugging, um sicherzustellen, dass die Werte jetzt korrekt sind
        print("✅ Debug: self.i2g_dict:", self.i2g_dict)  
        print("✅ Debug: self.g2i_dict:", self.g2i_dict)

        self.num_classes = num_classes
        self.search_mode = "max"
        self.blank_id = blank_id
        
        self.vocab =  ['|']+['-']+[chr(x) for x in range(20000, 20000 + num_classes)]

        # ✅ **Debugging: Überprüfe num_classes und Vokabular**
        #print(f"✅ Baseline num_classes: {self.num_classes}, Vokabulargröße: {len(self.vocab)}")
        #assert self.num_classes == len(self.vocab), "⚠️ Vokabulargröße passt nicht zu num_classes!"

        #print(f"✅ Blank ID: {self.blank_id}")
        #assert self.blank_id == 0, "⚠️ Blank ID ist nicht 0! Torchaudio erwartet meistens 0."

        print(f"✅ Tokens im CTC-Decoder: {self.vocab}")
        
        self.ctc_decoder = ctc_decoder(
            lexicon = None,
            tokens=self.vocab,
            beam_size=10
        )
        print(f"🔍 Debug: Verwende {self.ctc_decoder.__class__.__name__}")

        # 🚀 **Überprüfe das finale Vokabular in PyCTCDecode**
        print(f"✅ Torchaudio CTCDecoder verwendet: {self.vocab} (Größe: {len(self.vocab)})")


    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)
        
        
    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        decoder_outputs= self.ctc_decoder(nn_output, vid_lgt)
        
        # 🔍 Debugging: Überprüfe die Top-Wahrscheinlichkeiten und IDs aus den Logits
        top_probs, top_indices = nn_output.detach().max(-1)  # Nimmt die höchste Wahrscheinlichkeit pro Frame
        print(f"🔍 Max-Wahrscheinlichkeiten pro Frame:\n{top_probs.cpu().numpy()}")
        print(f"🔍 Höchstwahrscheinlich gewählte IDs:\n{top_indices.cpu().numpy()}")

        print(f"🔍 Rohe BeamSearch-Tokens: {decoder_outputs}")
        print(f"🔍 Zeitstempel pro Token: {[hyp.timesteps for batch in decoder_outputs for hyp in batch]}")
        print(f"🔍 Alle erkannten IDs (vor GroupBy): {top_indices.cpu().tolist()}")



        # 🔍 Debugging: Zeige die rohe Decodierung an
        print(f"🔍 Rohe Decodierung aus BeamSearch: {decoder_outputs}")

        tokens_per_batch = [[hyp.tokens for hyp in batch] for batch in decoder_outputs] # these are the TOP results

        print(f"🔍 Token-ID Mapping für BeamSearch:")
        for token in tokens_per_batch[0][0]:  # Iteriere über die vorhergesagten Token-IDs
            gloss = self.i2g_dict.get(int(token), f"UNK({token})")
            print(f"🔹 Token {token} → Gloss: {gloss}")

        timesteps_per_batch = [[hyp.timesteps for hyp in batch] for batch in decoder_outputs]

        # ✅ Debugging: Zeige Tokens pro Batch
        print(f"🔍 Tokens per Batch: {tokens_per_batch}")

        # 🔍 1️⃣ Zeige die maximalen Wahrscheinlichkeiten pro Frame
        max_probs, max_indices = nn_output.max(dim=-1)
        #print(f"🔍 Max-Wahrscheinlichkeiten pro Frame:\n{max_probs.numpy()}")
        #print(f"🔍 Höchstwahrscheinlich gewählte IDs:\n{max_indices.numpy()}")

        #score_per_batch = [[hyp.score for hyp in batch] for batch in beam_outputs]
        #words_per_batch = [[hyp.words for hyp in batch] for batch in beam_outputs]
        #print(f' beam results: {tokens_per_batch}')
        #print(f' timesteps: {timesteps_per_batch}')
        #print(f' beam scores: {score_per_batch}')
        #print(f' words: {words_per_batch}')

        # 🔍 2️⃣ Zeige rohe BeamSearch-Tokens
        print(f"🔍 Rohe BeamSearch-Tokens: {tokens_per_batch}")

        # 🔍 3️⃣ Zeige Timesteps der ausgewählten Tokens
        print(f"🔍 Timesteps per Batch: {timesteps_per_batch}")

        ret_list = []
        for batch_idx in range(len(tokens_per_batch)):  # Iterate over batches
            first_result = tokens_per_batch[batch_idx][0][:len(timesteps_per_batch[batch_idx][0])]

            print(f"🔍 first_result vor groupby: {first_result}, Typ: {type(first_result)}")
            print(f"🔍 Tokens vor GroupBy: {first_result.tolist()}")
            print(f"🔍 Mapped Glossen: {[self.i2g_dict.get(int(gloss_id), f'UNK({gloss_id})') for gloss_id in first_result.tolist()]}")


            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            
            ret_list.append([
                (self.i2g_dict.get(int(gloss_id), f"UNK({gloss_id})"), idx)
                for idx, gloss_id in enumerate(first_result)
            ])
        
        print(f"✅ Decoded sequence: {ret_list}")
        return ret_list
    

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        print(f'index_list.shape: {index_list.shape}')
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:int(vid_lgt[batch_idx].item())])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list

