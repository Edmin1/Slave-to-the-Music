from audiocraft.data.audio import audio_write
from audiocraft.models import JASCO

model = JASCO.get_pretrained(
    'facebook/jasco-chords-drums-melody-1B',
    chords_mapping_path='assets/chord_to_index_mapping.pkl'
)

model.set_generation_params(
    cfg_coef_all=1.5,
    cfg_coef_txt=0.5
)

text = "A classical orchestral piece with rich string harmonies and delicate piano passages."

output = model.generate_music(descriptions=[text], progress=True)
audio_write('JASCOSentenceClassical5', output.cpu().squeeze(0), model.sample_rate, strategy="loudness", loudness_compressor=True)