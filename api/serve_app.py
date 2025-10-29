import os
from ray import serve
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .sd_backend import SDBackend
from io import BytesIO
import base64

# FastAPI 앱 + Serve ingress
app = FastAPI(title="LoRA Inference API (Ray Serve)")

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

@serve.deployment(
    route_prefix="/",
    num_replicas=int(os.getenv("RAY_NUM_REPLICAS", "1")),
    ray_actor_options={"num_gpus": float(os.getenv("RAY_NUM_GPUS_PER_REPLICA", "1"))}
)
@serve.ingress(app)
class LoRAServeDeployment:
    def __init__(self):
        self.backend = SDBackend()

    @app.get("/healthz")
    def health(self):
        return {"status": "ok"}

    @app.post("/generate")
    def generate(self, req: GenRequest):
        try:
            if req.lora_repo and req.lora_weight:
                self.backend.attach_lora(req.lora_repo, req.lora_weight, req.lora_scale)
            img = self.backend.generate(req.prompt, steps=req.steps, guidance=req.guidance,
                                        seed=req.seed, size=(req.width, req.height))
            if req.lora_repo:
                self.backend.detach_lora()
            buf = BytesIO(); img.save(buf, format="PNG")
            return {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}
        except Exception as e:
            raise HTTPException(500, f"generation_failed: {e}")

# serve run api.serve_app:deployment 으로 실행
deployment = LoRAServeDeployment.bind()
