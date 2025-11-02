import os, torch, time
from diffusers import StableDiffusionPipeline

# Get base model from .env, if not exist use v1-5 as base model
BASE_MODEL = os.getenv("BASE_MODEL", "runwayml/stable-diffusion-v1-5")

def pick_device_dtype():
    # 가용한 디바이스와 dtype을 선택(우선순위: CUDA → MPS → CPU)
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32

class SDBackend:
    def __init__(self):
        # 초기화 
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
        # LoRA 어댑터 장착(요청 단위로 스타일/파인튜닝 효과 적용)
        # - repo_or_dir: HF repo id 또는 로컬 디렉토리
        # - weight_name: 사용할 LoRA 가중치 파일(.safetensors)
        # - scale: LoRA 영향도(1.0 기본, 높을수록 강함)
        # - adapter_name: 어댑터 식별자(멀티 어댑터 시 구분용)
        self.pipe.load_lora_weights(repo_or_dir, weight_name=weight_name, adapter_name=adapter_name)
        if hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters([adapter_name], adapter_weights=[float(scale)])

    def detach_lora(self):
        if hasattr(self.pipe, "unload_lora_weights"):
            self.pipe.unload_lora_weights()
        elif hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters([])

   
    def generate(self, prompt: str, steps: int = 22, guidance: float = 7.0, seed: int = 42, size=(512,512)):
        # 이미지 생성:
        # - prompt: 텍스트 프롬프트
        # - steps: 디퓨전 스텝(↑품질/↓속도)
        # - guidance: CFG 스케일(프롬프트 준수 강도)
        # - seed: 시드(재현성)
        # - size: (w,h) 해상도 — VRAM에 맞게 조절
        g = torch.Generator(device=self.device).manual_seed(int(seed))  # 디바이스별 난수생성기(재현성)
        w, h = size
        out = self.pipe(
            prompt,
            num_inference_steps=int(steps),      # 추론 스텝 수
            guidance_scale=float(guidance),      # CFG 스케일
            width=int(w), height=int(h),         # 출력 해상도
            generator=g                          # 고정 시드 생성기
        ).images[0]                               # 결과는 PIL.Image 리스트 → 첫 장 선택
        return out                                 # 호출자에게 PIL.Image 반환
