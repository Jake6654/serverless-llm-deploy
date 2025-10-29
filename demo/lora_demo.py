# lora_demo.py — 단일 프로세스 LoRA 교체 데모 (CUDA/MPS/CPU)

import os, time, torch
from diffusers import StableDiffusionPipeline
from PIL import Image

BASE_MODEL = os.getenv("BASE_MODEL", "runwayml/stable-diffusion-v1-5")

def pick_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32

def main():
    device, dtype = pick_device_dtype()
    print(f"[demo] device={device}, dtype={dtype}")

    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype, use_safetensors=True).to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = None
        except Exception:
            pass
    print(f"[demo] loaded in {time.time()-t0:.1f}s")

    # 프롬프트 & LoRA
    prompt = "a watercolor fox, soft light"
    lora_repo = "ybelkada/sd-1.5-pokemon-lora-peft"
    weight_name = "pytorch_lora_weights.safetensors"
    scale = 1.0

    # LoRA attach
    pipe.load_lora_weights(lora_repo, weight_name=weight_name, adapter_name="demo")
    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters(["demo"], adapter_weights=[scale])

    # 생성
    g = torch.Generator(device=device).manual_seed(123)
    img = pipe(prompt, num_inference_steps=20, guidance_scale=7.0, width=512, height=512, generator=g).images[0]

    # LoRA detach
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()
    elif hasattr(pipe, "set_adapters"):
        pipe.set_adapters([])

    os.makedirs("outputs", exist_ok=True)
    out = "outputs/lora_demo.png"
    img.save(out)
    print(f"[demo] saved: {out}")

if __name__ == "__main__":
    main()
