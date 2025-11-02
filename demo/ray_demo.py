# ray_demo.py — Ray-based parallel LoRA generation demo (CUDA/MPS/CPU 대응)

import os
import time
import ray
import torch
from sd_actor import DiffusionActorRemote  # same folder

os.environ.setdefault("RAY_DISABLE_CUSTOMIZED_TRACER", "1")

def init_ray():
    addr = os.getenv("RAY_ADDRESS")
    if addr:
        print(f"[Ray] Connecting to cluster: {addr}")
        ray.init(address=addr)
    else:
        print("[Ray] Starting local Ray runtime")
        ray.init()

def main():
    BASE_MODEL = os.getenv("BASE_MODEL", "runwayml/stable-diffusion-v1-5")
    DTYPE_STR = os.getenv("DTYPE_STR", "auto")

    gpu_per_actor = 1 if torch.cuda.is_available() else 0

    a1 = DiffusionActorRemote.options(num_gpus=gpu_per_actor).remote(BASE_MODEL, DTYPE_STR)
    a2 = DiffusionActorRemote.options(num_gpus=gpu_per_actor).remote(BASE_MODEL, DTYPE_STR)

    task1 = {
        "prompt": "a man working on his coding project, soft light, animation style",
        "lora_repo": "J-YOON/animate-lora-sd1.5",
        "weight_name": "animate_v1-000005.safetensors",
        "lora_scale": 1.0,
        "steps": 20,
        "guidance": 7.0,
        "seed": 123,
        "width": 512,
        "height": 512,
        "adapter_name": "anim",
    }
    task2 = {
        "prompt": "a watercolor fox, soft light, high detail",
        "lora_repo": "ybelkada/sd-1.5-pokemon-lora-peft",
        "weight_name": "pytorch_lora_weights.safetensors",
        "lora_scale": 1.0,
        "steps": 20,
        "guidance": 7.0,
        "seed": 456,
        "width": 512,
        "height": 512,
        "adapter_name": "poke",
    }

    t0 = time.time()
    futs = [
        a1.generate_with_lora.remote(**task1),
        a2.generate_with_lora.remote(**task2),
    ]
    (img1, gen1_time, act1), (img2, gen2_time, act2) = ray.get(futs)
    total = time.time() - t0

    os.makedirs("outputs", exist_ok=True)
    out1 = os.path.join("outputs", "ray_lora1_anim.png")
    out2 = os.path.join("outputs", "ray_lora2_pokemon.png")
    img1.save(out1)
    img2.save(out2)

    print("\n===== Summary =====")
    print(f"{act1} gen: {gen1_time:.2f}s")
    print(f"{act2} gen: {gen2_time:.2f}s")
    print(f"Parallel total: {total:.2f}s")
    print(f"Saved:\n - {out1}\n - {out2}")
    print("===================\n")

if __name__ == "__main__":
    init_ray()
    try:
        main()
    finally:
        ray.shutdown()
