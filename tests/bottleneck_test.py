# tests/bottleneck_test_auto.py
# ---------------------------------------------------------
# Stable Diffusion + LoRA Performance Bottleneck Test Script
# - Reuses the existing codebase (api/sd_backend.py)
# - Measures attach / generate / detach time separately
# - Runs multiple test cases automatically and exports results to CSV
# ---------------------------------------------------------

import os
import csv
import time
import argparse
from pathlib import Path
import sys

# Allow importing from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.sd_backend import SDBackend  # reuse existing backend logic


DEFAULT_PROMPT = "a cute animated girl, vibrant colors, clean outlines, studio lighting, upper body"
DEFAULT_LORA_REPO = os.getenv("TEST_LORA_REPO", "J-YOON/animate-lora-sd1.5")
DEFAULT_LORA_WEIGHT = os.getenv("TEST_LORA_WEIGHT", "animate_v1-000005.safetensors")


def now():
    """Returns high-resolution timestamp for precise timing."""
    return time.perf_counter()


def run_once(backend: SDBackend, *, use_lora: bool, cache_lora: bool, steps: int, width: int, height: int,
             prompt: str, lora_repo: str, lora_weight: str, lora_scale: float = 0.8):
    """
    Run a single inference and measure the time for LoRA attach / generate / detach.
    If cache_lora=True, assumes LoRA is already attached and skips attach/detach.
    """
    t_attach = 0.0
    t_detach = 0.0

    # 1) LoRA cold attach (only if not cached)
    if use_lora and not cache_lora:
        t0 = now()
        backend.attach_lora(lora_repo, lora_weight, lora_scale)
        t_attach = now() - t0

    # 2) Image generation
    t1 = now()
    _ = backend.generate(prompt, steps=steps, guidance=7.0, seed=123, size=(width, height))
    t_gen = now() - t1

    # 3) LoRA detach (only if not cached)
    if use_lora and not cache_lora:
        t2 = now()
        backend.detach_lora()
        t_detach = now() - t2

    return t_attach, t_gen, t_detach


def run_case(case: dict, *, repeats: int, prompt: str, lora_repo: str, lora_weight: str):
    """
    Run one test case multiple times and return average timings.
    If cache_lora=True, LoRA is attached once at start and detached once at end.
    """
    backend = SDBackend()  # fresh instance per case to isolate state

    # Pre-attach LoRA once if using cached mode
    pre_attach = 0.0
    post_detach = 0.0
    if case["use_lora"] and case["cache_lora"]:
        t0 = now()
        backend.attach_lora(lora_repo, lora_weight, case.get("scale", 0.8))
        pre_attach = now() - t0

    sum_attach = 0.0
    sum_gen = 0.0
    sum_detach = 0.0

    # Repeat several times for averaging
    for _ in range(repeats):
        a, g, d = run_once(
            backend,
            use_lora=case["use_lora"],
            cache_lora=case["cache_lora"],
            steps=case["steps"],
            width=case["width"],
            height=case["height"],
            prompt=prompt,
            lora_repo=lora_repo,
            lora_weight=lora_weight,
            lora_scale=case.get("scale", 0.8),
        )
        sum_attach += a
        sum_gen += g
        sum_detach += d

    # Detach LoRA after all repeats (for cached mode)
    if case["use_lora"] and case["cache_lora"]:
        t2 = now()
        backend.detach_lora()
        post_detach = now() - t2

    # Compute averages
    avg_attach = sum_attach / repeats
    avg_gen = sum_gen / repeats
    avg_detach = sum_detach / repeats

    return {
        "name": case["name"],
        "steps": case["steps"],
        "width": case["width"],
        "height": case["height"],
        "scale": case.get("scale", 0.8),
        "use_lora": case["use_lora"],
        "cache_lora": case["cache_lora"],
        "pre_attach": round(pre_attach, 4),
        "attach": round(avg_attach, 4),
        "gen": round(avg_gen, 4),
        "detach": round(avg_detach, 4),
        "post_detach": round(post_detach, 4),
        "total_per_iter": round(avg_attach + avg_gen + avg_detach, 4),
    }


def build_default_cases():
    """
    Test cases designed to clearly show differences:
    - A: LoRA attach overhead (cold / cached / none)
    - B: Step sweep (UNet iteration bottleneck)
    - C: Resolution sweep (spatial bottleneck) — changes both width & height
    """
    return [
        # A. LoRA attach overhead comparison
        {"name": "A_base_only",   "use_lora": False, "cache_lora": False, "steps": 22, "width": 512,  "height": 512,  "scale": 1.0},
        {"name": "A_lora_cold",   "use_lora": True,  "cache_lora": False, "steps": 22, "width": 512,  "height": 512,  "scale": 0.8},
        {"name": "A_lora_cached", "use_lora": True,  "cache_lora": True,  "steps": 22, "width": 512,  "height": 512,  "scale": 0.8},

        # B. Step sweep (linear scaling)
        {"name": "B_steps_10",    "use_lora": True,  "cache_lora": True,  "steps": 10, "width": 512,  "height": 512,  "scale": 0.8},
        {"name": "B_steps_22",    "use_lora": True,  "cache_lora": True,  "steps": 22, "width": 512,  "height": 512,  "scale": 0.8},
        {"name": "B_steps_50",    "use_lora": True,  "cache_lora": True,  "steps": 50, "width": 512,  "height": 512,  "scale": 0.8},

        # C. Resolution sweep (quadratic scaling)
        {"name": "C_res_512",     "use_lora": True,  "cache_lora": True,  "steps": 22, "width": 512,  "height": 512,  "scale": 0.8},
        {"name": "C_res_768",     "use_lora": True,  "cache_lora": True,  "steps": 22, "width": 768,  "height": 768,  "scale": 0.8},
        {"name": "C_res_1024",    "use_lora": True,  "cache_lora": True,  "steps": 22, "width": 1024, "height": 1024, "scale": 0.8},
    ]


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion + LoRA Bottleneck Auto Test")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--lora-repo", default=DEFAULT_LORA_REPO)
    parser.add_argument("--lora-weight", default=DEFAULT_LORA_WEIGHT)
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions per case (average over)")
    parser.add_argument("--out", default="bottleneck_results.csv")
    args = parser.parse_args()

    cases = build_default_cases()
    results = []

    print(f"[INFO] Running {len(cases)} cases x {args.repeats} repeats")
    print(f"[INFO] LoRA: {args.lora_repo} :: {args.lora_weight}")
    print("-" * 72)

    for case in cases:
        print(f"-> Case: {case['name']} (steps={case['steps']}, size={case['width']}x{case['height']}, "
              f"lora={'on' if case['use_lora'] else 'off'}, cache={case['cache_lora']})")

        res = run_case(case, repeats=args.repeats, prompt=args.prompt,
                       lora_repo=args.lora_repo, lora_weight=args.lora_weight)
        results.append(res)

        print(f"   pre_attach={res['pre_attach']:.3f}s, attach={res['attach']:.3f}s, "
              f"gen={res['gen']:.3f}s, detach={res['detach']:.3f}s, post_detach={res['post_detach']:.3f}s, "
              f"total/iter={res['total_per_iter']:.3f}s")

    # Save results to CSV
    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("-" * 72)
    print(f"[DONE] Results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
