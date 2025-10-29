# ray_demo.py — Ray-based parallel LoRA generation demo (MPS/CPU friendly)

import os
os.environ.setdefault("RAY_DISABLE_CUSTOMIZED_TRACER", "1")  # (선택) 직렬화 이슈 완화
import time
import ray
from sd_actor import DiffusionActorRemote  # 모듈에서 Actor 불러오기

if __name__ == "__main__":
    BASE_MODEL = "runwayml/stable-diffusion-v1-5"
    DTYPE_STR = "float16"  # MPS면 float16 권장

    # 작업 1
    task1 = {
        "prompt": "a man working on his coding project, soft light, animation style",
        "lora_repo": "J-YOON/animate-lora-sd1.5",
        "lora_weight": "animate_v1-000005.safetensors",
        "lora_scale": 1.0,
        "seed": 42,
        "out_name": "ray_lora1_animate.png",
    }
    # 작업 2
    task2 = {
    "prompt": "a pikachu working on his coding project, soft light",
    "lora_repo": "ybelkada/sd-1.5-pokemon-lora-peft",   # ← 점(.) 사용!
    "lora_weight": "pytorch_lora_weights.safetensors",   # 파일명 일치
    "lora_scale": 1.0,
    "seed": 123,
    "out_name": "ray_lora2_pokemon.png",
}

    print("Ray 초기화...")
    ray.init()

    num_actors = 2
    if ray.cluster_resources().get("GPU", 0) < num_actors:
        print(f"경고: CUDA GPU {num_actors}개 필요하지만 감지 {ray.cluster_resources().get('GPU', 0)}개 (Mac/MPS 정상)")

    print(f"{num_actors}개의 DiffusionActor 생성...")
    actors = [DiffusionActorRemote.remote(BASE_MODEL, DTYPE_STR) for _ in range(num_actors)]

    print("작업 병렬 제출...")
    needed = ["prompt", "lora_repo", "lora_weight", "lora_scale", "seed"]  # 필요한 키만 전달
    job1_ref = actors[0].generate_with_lora.remote(**{k: task1[k] for k in needed})
    job2_ref = actors[1].generate_with_lora.remote(**{k: task2[k] for k in needed})

    t0 = time.time()
    results = ray.get([job1_ref, job2_ref])
    total_time = time.time() - t0

    (img1, gen1_time, a1) = results[0]
    (img2, gen2_time, a2) = results[1]

    os.makedirs("outputs", exist_ok=True)
    img1.save(os.path.join("outputs", task1["out_name"]))
    img2.save(os.path.join("outputs", task2["out_name"]))

    print("\n===== Summary =====")
    print(f"Actor1({a1}) gen: {gen1_time:.2f}s")
    print(f"Actor2({a2}) gen: {gen2_time:.2f}s")
    print(f"Parallel total: {total_time:.2f}s")
    print(f"Saved: outputs/{task1['out_name']}, outputs/{task2['out_name']}")
    print("===================\n")

    ray.shutdown()
