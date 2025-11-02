# sd_actor.py
import os
import time
import torch
import ray

class DiffusionActor:
    """
    pre-load Stable Diffusion pipline,
    whenever request comes, Actor do  LoRA attach -> create image -> LoRA detach.
    - Device Priority: cuda > mps > cpu
    - dtype: cuda/mps(float16) / cpu(float32)
    """
    def __init__(self, base_model_id: str = "runwayml/stable-diffusion-v1-5", dtype_str: str = "auto"):
        from diffusers import StableDiffusionPipeline  # lazy import

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if dtype_str == "auto":
            self.dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        else:
            self.dtype = getattr(torch, dtype_str)

        t0 = time.time()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        ).to(self.device)

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        if hasattr(self.pipe, "safety_checker"):
            try:
                self.pipe.safety_checker = None
            except Exception:
                pass

        print(f"[Actor:{os.getpid()}] Loaded {base_model_id} on {self.device} ({self.dtype}) in {time.time()-t0:.1f}s")

    def generate_with_lora(
        self,
        prompt: str,
        lora_repo: str | None = None,
        weight_name: str | None = None,
        lora_scale: float = 1.0,
        steps: int = 22,
        guidance: float = 7.0,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
        adapter_name: str = "current",
    ):
        # 1) attach (요청 시에만)
        if lora_repo and weight_name:
            self.pipe.load_lora_weights(lora_repo, weight_name=weight_name, adapter_name=adapter_name)
            if hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters([adapter_name], adapter_weights=[float(lora_scale)])

        # 2) generate
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        t_gen = time.time()
        image = self.pipe(
            prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            generator=g,
        ).images[0]
        gen_time = time.time() - t_gen

        # 3) detach
        if lora_repo and weight_name:
            if hasattr(self.pipe, "unload_lora_weights"):
                self.pipe.unload_lora_weights()
            elif hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters([])

        return image, gen_time, f"actor_{os.getpid()}"

DiffusionActorRemote = ray.remote(DiffusionActor)
