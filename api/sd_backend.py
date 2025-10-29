import os, torch, time
from diffusers import StableDiffusionPipeline

BASE_MODEL = os.getenv("BASE_MODEL", "runwayml/stable-diffusion-v1-5")

def pick_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32

class SDBackend:
    def __init__(self):
        self.device, self.dtype = pick_device_dtype()
        t0 = time.time()
        self.pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=self.dtype, use_safetensors=True).to(self.device)
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        if hasattr(self.pipe, "safety_checker"):
            try:
                self.pipe.safety_checker = None
            except Exception:
                pass
        print(f"[SDBackend] Loaded {BASE_MODEL} on {self.device}/{self.dtype} in {time.time()-t0:.1f}s")

    def attach_lora(self, repo_or_dir: str, weight_name: str, scale: float = 1.0, adapter_name="current"):
        self.pipe.load_lora_weights(repo_or_dir, weight_name=weight_name, adapter_name=adapter_name)
        if hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters([adapter_name], adapter_weights=[float(scale)])

    def detach_lora(self):
        if hasattr(self.pipe, "unload_lora_weights"):
            self.pipe.unload_lora_weights()
        elif hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters([])

    def generate(self, prompt: str, steps: int = 22, guidance: float = 7.0, seed: int = 42, size=(512,512)):
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        w, h = size
        out = self.pipe(prompt, num_inference_steps=int(steps), guidance_scale=float(guidance),
                        width=int(w), height=int(h), generator=g).images[0]
        return out
