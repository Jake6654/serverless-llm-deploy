import os
from ray import serve
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .sd_backend import SDBackend
from io import BytesIO
import base64

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

# (선택) 오토스케일 값은 환경변수로
autoscaling = {
    "min_replicas": int(os.getenv("SERVE_MIN_REPLICAS", "1")),
    "max_replicas": int(os.getenv("SERVE_MAX_REPLICAS", "1")),
    "target_num_ongoing_requests_per_replica": int(os.getenv("SERVE_TARGET_QPS_PER_REPLICA", "2")),
}

@serve.deployment(
    autoscaling_config=autoscaling,
    ray_actor_options={"num_gpus": float(os.getenv("RAY_NUM_GPUS_PER_REPLICA", "1"))},
    name="LoRAServeDeployment",
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
            img = self.backend.generate(
                req.prompt, steps=req.steps, guidance=req.guidance,
                seed=req.seed, size=(req.width, req.height)
            )
            if req.lora_repo:
                self.backend.detach_lora()
            buf = BytesIO(); img.save(buf, format="PNG")
            return {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}
        except Exception as e:
            raise HTTPException(500, f"generation_failed: {e}")

# 배인딩
deployment = LoRAServeDeployment.bind()
