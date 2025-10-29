# sd_actor.py
import os
import time
import torch
import ray

class DiffusionActor:
    """
    Stable Diffusion 파이프라인을 미리 로드하고,
    요청마다 LoRA attach -> 이미지 생성 -> LoRA detach 를 수행하는 Actor.
    """
    def __init__(self, base_model_id, dtype_str):
        # ❗️지연 import (전역에서 diffusers import 금지)
        from diffusers import StableDiffusionPipeline

        pid = os.getpid()
        print(f"[Actor {pid}] start. Base model loading...")

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float16 if dtype_str == "float16" else torch.float32

        t0 = time.time()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id, torch_dtype=self.dtype
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print(f"[Actor {pid}] load complete ({time.time() - t0:.2f}s)")

    # 반환 타입 힌트 없음 (Ray 직렬화 안전)
    def generate_with_lora(self, prompt, lora_repo, lora_weight, lora_scale, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)

        # 1) LoRA attach
        t_attach = time.time()
        self.pipe.load_lora_weights(
            lora_repo, weight_name=lora_weight, adapter_name="current_lora"
        )
        if hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters(["current_lora"], adapter_weights=[float(lora_scale)])
        print(f"[Actor {os.getpid()}] LoRA attach OK ({time.time() - t_attach:.2f}s)")

        # 2) 생성
        t_gen = time.time()
        img = self.pipe(
            prompt, num_inference_steps=22, guidance_scale=7.0, generator=g
        ).images[0]
        gen_time = time.time() - t_gen

        # 3) LoRA detach
        if hasattr(self.pipe, "unload_lora_weights"):
            self.pipe.unload_lora_weights()
        elif hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters([])

        return img, gen_time, f"actor_{os.getpid()}"

# 모듈 레벨에서 post-remote 래핑
DiffusionActorRemote = ray.remote(num_cpus=1)(DiffusionActor)
