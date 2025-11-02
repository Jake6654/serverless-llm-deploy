# local test file

from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel, Field
from io import BytesIO # memory buffer image to byte
import base64 # byte -> base64 encoding
from .sd_backend import SDBackend 

app = FastAPI(title="LoRA Inference API") # FastAPI instance 
backend = SDBackend()  # load just once since it's heavy task

class GenRequest(BaseModel):
    # 요청 JSON 스키마 정의 
    prompt: str
    lora_repo: str | None = None
    lora_weight: str | None = None
    lora_scale: float = Field(1.0, ge=0.0)
    steps: int = Field(22, ge=5, le=75) # 디퓨전 스텝 수 (quality/speed trade off)
    guidance: float = Field(7.0, ge=0.0, le=20.0)
    seed: int = 42
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)

@app.get("/healthz") # health check endpoint
def health():
    return {"status": "ok"}

@app.post("/generate") # image generation endpoint
def generate(req: GenRequest):
    try:
        # only when lora assign then attach an adapter
        if req.lora_repo and req.lora_weight:
            backend.attach_lora(req.lora_repo, req.lora_weight, req.lora_scale)
        # call stable diffusion 
        img = backend.generate(req.prompt, steps=req.steps, guidance=req.guidance,
                               seed=req.seed, size=(req.width, req.height))
        if req.lora_repo:
            backend.detach_lora()
        # PIL.Image -> memory buffer -> Base64 string return
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # JSON 으로 Base64 PNG 를 반환 (클라이언트를 디코딩해 이미지로 사용 )
        return {"image_base64": b64}
    except Exception as e:
        raise HTTPException(500, f"generation_failed: {e}")
