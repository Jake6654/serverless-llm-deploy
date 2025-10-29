from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from io import BytesIO
import base64
from .sd_backend import SDBackend

app = FastAPI(title="LoRA Inference API")
backend = SDBackend()  # 프로세스당 1회 로드

class GenRequest(BaseModel):
    prompt: str
    lora_repo: str | None = None
    lora_weight: str | None = None
    lora_scale: float = Field(1.0, ge=0.0)
    steps: int = Field(22, ge=5, le=75)
    guidance: float = Field(7.0, ge=0.0, le=20.0)
    seed: int = 42
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenRequest):
    try:
        if req.lora_repo and req.lora_weight:
            backend.attach_lora(req.lora_repo, req.lora_weight, req.lora_scale)
        img = backend.generate(req.prompt, steps=req.steps, guidance=req.guidance,
                               seed=req.seed, size=(req.width, req.height))
        if req.lora_repo:
            backend.detach_lora()

        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image_base64": b64}
    except Exception as e:
        raise HTTPException(500, f"generation_failed: {e}")
