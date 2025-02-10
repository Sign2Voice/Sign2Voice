import torch
from utils.decode import Decode

# 🏗 **Correct Test Gloss Dictionary with Numerical Indices**
gloss_dict = {
    "WEATHER": [0, 100],
    "SUN": [1, 80],
    "RAIN": [2, 60],
    "WIND": [3, 50],
    "COLD": [4, 40],
    "WARM": [5, 30]
}  # 🎯 Format matches actual training

# 🧠 **Example Logits for a Sequence of 8 Frames with 8 Classes**
# The values must be compatible with `self.vocab` (['|', '-', 'U20000', 'U20001', ...])
logits = torch.tensor([
    [0.1, 0.6, 0.1, 0.1, 0.1, 0.0],  # Frame 1 → SUN
    [0.1, 0.1, 0.7, 0.1, 0.0, 0.0],  # Frame 2 → RAIN
    [0.1, 0.1, 0.1, 0.6, 0.1, 0.0],  # Frame 3 → WIND
    [0.7, 0.1, 0.1, 0.1, 0.0, 0.0],  # Frame 4 → WEATHER
    [0.1, 0.1, 0.1, 0.1, 0.6, 0.0],  # Frame 5 → COLD
    [0.1, 0.6, 0.1, 0.1, 0.1, 0.0],  # Frame 6 → SUN (Repetition for Testing)
    [0.1, 0.1, 0.7, 0.1, 0.0, 0.0],  # Frame 7 → RAIN
    [0.1, 0.1, 0.1, 0.6, 0.1, 0.0],  # Frame 8 → WIND
])  # Format (T, N) (8 Frames, 6 Classes including CTC-Blank)

# 🎯 **Initialize Decoder with Correct num_classes**
num_classes = len(gloss_dict) + 2  # `+2` for "|" (Blank) and "-" (Pause)
decoder = Decode(gloss_dict, num_classes=num_classes, search_mode="beam")

# 📏 **Dummy Video Length**
vid_lgt = torch.tensor([logits.shape[0]])  # Sequence Length = 8 Frames

# 🔄 **Decoding**
decoded = decoder.decode(logits.unsqueeze(0), vid_lgt, batch_first=True, probs=False)

print(f"🔍 Logits for Batch 0:\n{logits.numpy()}")

# ✅ **Display Result**
print("\n🔍 **Result of Frame Sequence:**")
print("Decoded sequence:", decoded)

# 🚨 **Check if 'UNKNOWN' Appears**
contains_unknown = any("UNKNOWN" in gloss for gloss, _ in decoded[0])
if contains_unknown:
    print("🚨 WARNING: 'UNKNOWN' gloss found!")
else:
    print("✅ No 'UNKNOWN' gloss found! Everything mapped correctly!")







