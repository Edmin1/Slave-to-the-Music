import torch
import torchaudio
from hear21passt.base import get_basic_model
import argparse
import os
import warnings
import csv
import re

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=FutureWarning, module="hear21passt.models.preprocess")
warnings.filterwarnings("ignore", category=UserWarning, module="hear21passt.models.passt")

def load_class_mapping(csv_path="class_labels_indices.csv"):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return {int(row["index"]): row["display_name"] for row in reader}

def get_allowed_indices(filter_type, class_map):
    genre_labels = {
        "Pop music", "Hip hop music", "Rock music", "Rhythm and blues", "Soul music",
        "Reggae", "Country", "Funk", "Folk music", "Jazz", "Disco", "Classical music",
        "Electronic music", "Blues", "New-age music", "Ska"
    }

    instrument_keywords = {
        "guitar", "drum", "piano", "saxophone", "violin", "cello", "banjo", "bass",
        "synth", "keyboard", "flute", "harp", "trumpet", "trombone", "clarinet",
        "xylophone", "accordion", "harmonica", "bagpipes", "organ", "instrument"
    }

    allowed = set()
    for idx, label in class_map.items():
        label_clean = label.strip()
        if filter_type == "genres" and label_clean in genre_labels:
            allowed.add(idx)
        elif filter_type == "music":
            if label_clean in genre_labels or any(instr in label_clean.lower() for instr in instrument_keywords):
                allowed.add(idx)

    return allowed

def load_audio(file_path, target_sr=32000):
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    target_length = 998 * 32
    if waveform.shape[1] > target_length:
        start = (waveform.shape[1] - target_length) // 2
        waveform = waveform[:, start:start + target_length]
    elif waveform.shape[1] < target_length:
        pad_amt = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
    return waveform

def classify_audio(audio_path, class_map, filter_type=None, threshold=0.05):
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return

    print(f"Processing file: {audio_path}")
    model = get_basic_model(mode="logits").eval()
    waveform = load_audio(audio_path)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    waveform = waveform.to(device)

    with torch.no_grad():
        logits = model(waveform)
        probs = torch.sigmoid(logits)[0]

    allowed = get_allowed_indices(filter_type, class_map) if filter_type else None

    indices = []
    probs_above = []

    for idx, prob in enumerate(probs):
        if prob > threshold and (not allowed or idx in allowed):
            indices.append(idx)
            probs_above.append(prob)

    if not indices:
        print(f"\nNo predictions above {threshold*100:.0f}% confidence")
        return

    sorted_items = sorted(zip(indices, probs_above), key=lambda x: x[1], reverse=True)
    print(f"\nPredictions above {threshold*100:.0f}% confidence:")
    for idx, prob in sorted_items[:100]:  # show top 5
        label = class_map.get(idx, f"Unknown {idx}")
        print(f"{label} (Index {idx}): {prob:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify audio using PaSST')
    parser.add_argument("audio_path", type=str, help="Path to audio file (wav/mp3)")
    parser.add_argument("--filter", type=str, choices=["music", "genres"], help="Limit results")
    parser.add_argument("--threshold", type=float, default=0.000000001, help="Confidence threshold")
    args = parser.parse_args()

    class_map = load_class_mapping("class_labels_indices.csv")
    classify_audio(args.audio_path, class_map, args.filter, args.threshold)
