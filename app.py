# app.py  (sanity + generation with loud prints)
import sys, time, os
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype  = torch.float16 if device == "mps" else torch.float32
print(f"🚀 Using device={device}, dtype={dtype}")

# check if diffusers is properly imported
try:
    from diffusers import StableDiffusionPipeline
    print("📦 diffusers import OK")
except Exception as e:
    print("❌ diffusers import failed:", repr(e))
    sys.exit(1)

print("⬇️  Loading base model (first time can take several minutes)...", flush=True)
t0 = time.time()
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)
# progress bar to debug
pipe.set_progress_bar_config(disable=False)
print(f"✅ Model loaded in {time.time()-t0:.1f}s")


prompt = "a watercolor fox, 512x512, soft light"
print("🎨 Generating image:", prompt, flush=True)
t1 = time.time()
img = pipe(prompt, num_inference_steps=20, guidance_scale=7.0).images[0]


img.save("output.png")
print(f"🖼️  Saved output.png  (gen {time.time()-t1:.1f}s)")
