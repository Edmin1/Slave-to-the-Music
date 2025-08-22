import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model, model_config = get_pretrained_model("HKUSTAudio/AudioX")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

seconds_start = 0
seconds_total = 10
model = model.to(device)

text_prompt = "A classical orchestral piece with rich string harmonies and delicate piano passages."

conditioning = [{
    "text_prompt": text_prompt,
    "video_prompt": None,
    "audio_prompt": None,
    "seconds_start": 0,
    "seconds_total": 10
}]
output = generate_diffusion_cond(
    model,
    steps=250,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

output = rearrange(output, "b d n -> d (b n)")

output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("AudioXClassical5.wav", output, sample_rate)
