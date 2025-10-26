import os
import time
import argparse
import torch
from diffusers import StableDiffusionPipeline


import os
import time
import torch
from diffusers import StableDiffusionPipeline

# -------------------- Configuration (you can edit these directly) --------------------
base_model = "runwayml/stable-diffusion-v1-5"   # Base SD model
prompt = "a man working on his coding project, soft light, 512x512"  # Text prompt
steps = 22                      # Number of diffusion steps
guidance = 7.0                  # CFG scale (creativity)
seed = 42                       # Random seed


device = "mps" if torch.backends.mps.is_available() else "cpu" # use mps if it is available
dtype = torch.float16 if device == "mps" else torch.float32 #

# LoRA options 
lora_repo = "J-YOON/animate-lora-sd1.5"         # Hugging Face repo or local folder
lora_weight = "animate_v1-000005.safetensors"   # File name inside that repo/folder
lora_scale = 1.5                                # How strong the LoRA effect is

lora_repo2 ="ybelkada/sd-1.5-pokemon-lora-peft"
lora_weight2 = "pytorch_lora_weights.safetensors"
outdir = "outputs"                              # Save images here

# -------------------- Setup --------------------
print(f"Using device={device}, dtype={dtype}")
os.makedirs(outdir, exist_ok=True)
g = torch.Generator(device=device).manual_seed(seed)

# -------------------- Load base pipeline --------------------
print(f"⬇️  Loading base model: {base_model}")
t0 = time.time()
pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype).to(device)
pipe.set_progress_bar_config(disable=False)
print(f"✅ Base model ready in {time.time() - t0:.1f}s")

# -------------------- LoRA helpers --------------------
def _pipeline_base_id(p):
    name = getattr(p, "_name_or_path", None)
    if name:
        return name
    cfg = getattr(p, "config", None)
    if isinstance(cfg, dict):
        return cfg.get("_name_or_path")
    if cfg is not None:
        return getattr(cfg, "_name_or_path", None)
    return None

def detach_lora(p):
    """Detach/disable any active LoRA in a version-compatible way."""
    if hasattr(p, "unload_lora_weights"):
        p.unload_lora_weights()
        return "unload_lora_weights()"
    # Clears all active LoRA adapters
    if hasattr(p, "set_adapters"):
        try:
            p.set_adapters([])
            return "set_adapters([])"
        except Exception:
            pass
    raise RuntimeError("Could not detach LoRA. Try reloading the pipeline.")

def attach_lora(p, lora_id_or_path, scale=1.0, weight_name=None):
    """Attach a LoRA (local or Hugging Face repo)."""
    _weight_name = weight_name
    lora_dir = lora_id_or_path

    # pass LoRA API
    if hasattr(p, "load_lora_weights"):
        if _weight_name:
            p.load_lora_weights(lora_dir, weight_name=_weight_name)
        else:
            p.load_lora_weights(lora_dir)
        
        
        if hasattr(p, "set_adapters"):
            try:
                p.set_adapters(["default"], adapter_weights=[float(scale)])
            except Exception:
                pass
        # 없으면 가중치를 합성
        elif hasattr(p, "fuse_lora"):
            try:
                p.fuse_lora(lora_scale=float(scale))
            except Exception:
                pass
        return "load_lora_weights()"

    raise RuntimeError("This diffusers version does not expose a LoRA loading API.")

# -------------------- Generation helper --------------------
def generate_once(prompt, steps, guidance, fname):
    """Run one inference and save the resulting PNG."""
    t_start = time.time()
    result = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=g)
    img = result.images[0]
    out_path = os.path.join(outdir, fname)
    img.save(out_path)
    secs = time.time() - t_start
    print(f"🖼️  Saved {out_path}  (gen {secs:.1f}s)")
    return secs

# -------------------- 1) Base-only (cold) --------------------
print(f"lora_repo2:{lora_repo2}" )
print("[1/4] BASE ONLY (cold)")
base_secs_cold = generate_once(prompt, steps, guidance, "base.png")

# -------------------- 2) With LoRA --------------------
lora_secs = None
attach_secs = None
if lora_repo:
    print(f"Attaching LoRA: {lora_repo}  (scale={lora_scale})")
    t_attach = time.time()
    method_used = attach_lora(pipe, lora_repo, scale=lora_scale, weight_name=lora_weight)
    attach_secs = time.time() - t_attach
    print(f"⚙️  LoRA attached via {method_used} in {attach_secs:.2f}s")
    print("🎨 [2/4] WITH LoRA")
    lora_secs = generate_once(prompt, steps, guidance, "lora.png")
else:
    print("No LoRA provided; skipping LoRA generation.")

# -------------------- 3) Detach and generate again --------------------
print("🧹 Detaching any LoRA (back to base-only)")
method_used = detach_lora(pipe)
print(f"LoRA detached via {method_used}")
print("[3/4] BASE (warm)")
base_secs_warm = generate_once(prompt, steps, guidance, "base_warm.png")

# 4 attach a different LoRA

if lora_repo2:
    print(f"Attaching LoRA: {lora_repo2}  (scale={lora_scale})")
    t_attach = time.time()
    method_used = attach_lora(pipe, lora_repo2, scale=lora_scale, weight_name=lora_weight2)
    attach_secs = time.time() - t_attach
    print(f"⚙️  LoRA attached via {method_used} in {attach_secs:.2f}s")
    print("🎨 [4/4] WITH LoRA")
    lora_secs = generate_once(prompt, steps, guidance, "lora2.png")
else:
    print("No LoRA provided; skipping LoRA generation.")


# -------------------- Summary --------------------
print("\n===== Summary =====")
print(f"Device: {device} | Steps: {steps} | Guidance: {guidance} | Seed: {seed}")
print(f"Base (cold): {base_secs_cold:.1f}s")
if lora_secs is not None:
    print(f"LoRA attach: {attach_secs:.2f}s")
    print(f"LoRA gen:    {lora_secs:.1f}s")
print(f"Base (warm): {base_secs_warm:.1f}s")
print("===================\n")
