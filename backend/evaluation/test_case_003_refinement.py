"""
Step 11N: Representative Real VLM inference on case_003_vegetation
with the refined Optical-SAR specialist prompt.
"""
from pathlib import Path
from PIL import Image
import os
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.vlm.optical_sar import build_optical_sar_prompt
from app.vlm.model import VLM

vis_dir = BACKEND_DIR / "evaluation" / "optical_sar" / "visualizations" / "case_003_vegetation"
opt_p = vis_dir / "case_003_vegetation_optical.png"
vv_p = vis_dir / "case_003_vegetation_sar_vv.png"
vh_p = vis_dir / "case_003_vegetation_sar_vh.png"
comp_p = vis_dir / "case_003_vegetation_sar_composite.png"

question = (
    "Describe the dominant vegetation patterns using both optical and SAR evidence. "
    "Explain what the optical image suggests, what VV and VH suggest, and how the modalities complement each other. "
    "Clearly distinguish direct observations from inference and state any uncertainty."
)

prompt = build_optical_sar_prompt(
    question=question,
    optical_metadata={"is_false_color": False, "description": "True-color RGB (B04, B03, B02)"},
    available_sar_modalities=["sar_vv", "sar_vh", "sar_composite"],
)

if not os.getenv("HF_TOKEN"):
    print("[STATUS: HF_TOKEN NOT SET IN CURRENT SHELL]")
    print("To run with live VLM, execute in your active terminal:")
    print("  python evaluation/test_case_003_refinement.py")
    sys.exit(0)

v = VLM()
opt = Image.open(opt_p)
vv = Image.open(vv_p)
vh = Image.open(vh_p)
comp = Image.open(comp_p)

answer = v.generate(
    image=opt,
    question=prompt,
    images={"s1_vv": vv, "s1_vh": vh, "s1_composite": comp},
)
print("=== REFINED VLM RESPONSE ===")
print(answer)
