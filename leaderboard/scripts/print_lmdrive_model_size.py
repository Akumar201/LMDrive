#!/usr/bin/env python3
"""
Print LMDrive model parameter counts and estimated GPU memory per component.
Run from LMDrive project root with the same env as evaluation (e.g. conda activate lmdrive).

  cd /home/akumar/LMDrive
  PYTHONPATH=.:leaderboard:leaderboard/team_code:LAVIS:vision_encoder python3 leaderboard/scripts/print_lmdrive_model_size.py

Or use the shell wrapper: ./leaderboard/scripts/print_model_size.sh
"""
import os
import sys

# Ensure we can import leaderboard config and LAVIS
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for sub in ("", "leaderboard", "leaderboard/team_code", "LAVIS", "vision_encoder"):
    path = os.path.join(ROOT, sub) if sub else ROOT
    if path not in sys.path:
        sys.path.insert(0, path)

def count_params(module):
    return sum(p.numel() for p in module.parameters())

def count_buffers(module):
    return sum(b.numel() for b in module.buffers())

def main():
    import torch
    import imp

    # Load config (same as lmdriver_agent)
    config_path = os.path.join(ROOT, "leaderboard", "team_code", "lmdriver_config.py")
    if not os.path.isfile(config_path):
        print("Config not found:", config_path)
        sys.exit(1)
    config = imp.load_source("MainModel", config_path).GlobalConfig()

    from lavis.common.registry import registry
    from timm.models import create_model

    print("Building LMDrive model (this loads the 7B LLM and may take a minute)...")
    model_cls = registry.get_model_class("vicuna_drive")

    # Resolve paths relative to project root
    def abspath(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)
    vision_ckpt = abspath(getattr(config, "preception_model_ckpt", ""))
    load_vision = vision_ckpt and os.path.isfile(vision_ckpt)

    # Build model (vision encoder loads from ckpt if present)
    model = model_cls(
        preception_model=config.preception_model,
        preception_model_ckpt=vision_ckpt if load_vision else "",
        load_pretrained=load_vision,
        llm_model=config.llm_model,
        max_txt_len=64,
        use_notice_prompt=getattr(config, "agent_use_notice", False),
    )

    # Optional: load LMDrive checkpoint (adds no new params, just weights)
    ckpt_path = abspath(getattr(config, "lmdrive_ckpt", "") or "")
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
            print("Loaded LMDrive checkpoint:", ckpt_path)

    # Component param counts
    components = [
        ("LLM (llava-v1.5-7b)", model.llm_model),
        ("Visual encoder (MemFuser R50 e1d3)", model.visual_encoder),
        ("ln_vision", model.ln_vision),
        ("Q-Former", model.Qformer if getattr(model, "Qformer", None) is not None else None),
        ("query_tokens", None),  # part of params, counted inside Qformer or separately
        ("llm_proj", model.llm_proj),
        ("waypoints_predictor", model.waypoints_predictor),
        ("waypoints_fc", getattr(model, "waypoints_fc", None)),
        ("waypoints_output", getattr(model, "waypoints_output", None)),
        ("end_predictor", model.end_predictor),
    ]

    total_params = 0
    total_buffers = 0
    rows = []

    for name, module in components:
        if module is None:
            continue
        nparam = count_params(module)
        nbuf = count_buffers(module)
        total_params += nparam
        total_buffers += nbuf
        # Estimate GPU memory: fp16 = 2 bytes/param, buffers often fp32 = 4 bytes
        mem_mb = (nparam * 2 + nbuf * 4) / (1024 ** 2)
        rows.append((name, nparam, nbuf, mem_mb))

    # query_tokens is a Parameter in the model; if not inside Qformer, count it
    if hasattr(model, "query_tokens") and model.query_tokens is not None:
        nq = model.query_tokens.numel()
        total_params += nq
        mem_mb = (nq * 2) / (1024 ** 2)
        rows.append(("query_tokens", nq, 0, mem_mb))

    # Sort by param count descending
    rows.sort(key=lambda x: -x[1])

    print()
    print("=" * 80)
    print("LMDrive model size breakdown (weights only; activations add more at runtime)")
    print("=" * 80)
    print(f"{'Component':<45} {'Params':>12} {'Buffers':>10} {'Est. GPU (MB)':>14}")
    print("-" * 80)
    for name, nparam, nbuf, mem_mb in rows:
        print(f"{name:<45} {nparam:>12,} {nbuf:>10,} {mem_mb:>13.1f}")
    print("-" * 80)
    total_mem_mb = (total_params * 2 + total_buffers * 4) / (1024 ** 2)
    print(f"{'TOTAL':<45} {total_params:>12,} {total_buffers:>10,} {total_mem_mb:>13.1f}")
    print("=" * 80)
    print()
    print("Notes:")
    print("  - LLM dominates (~7B params ≈ 14 GB in fp16).")
    print("  - Visual encoder and Q-Former are small by comparison.")
    print("  - Runtime activations (especially in decoder attention) need extra VRAM.")
    print("  - Your GPU: 16 GB; close other GPU apps or use a smaller LLM to avoid OOM.")
    print()

if __name__ == "__main__":
    main()
