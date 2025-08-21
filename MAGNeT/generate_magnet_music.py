from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

model = MAGNeT.get_pretrained("facebook/magnet-medium-30secs")

descriptions = ["A classical music piece with a grand piano and strings."]

wav_outputs = model.generate(descriptions)

for idx, wav in enumerate(wav_outputs):
    audio_write(f"MAGNeTClassicaltest", wav.cpu(), model.sample_rate, strategy="loudness")