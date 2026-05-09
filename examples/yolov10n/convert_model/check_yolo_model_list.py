from pathlib import Path
import yaml
import ultralytics

root = Path(ultralytics.__file__).resolve().parent / "cfg" / "models"

print("models dir:", root)

for f in sorted(root.rglob("*.yaml")):
    with open(f, "r", encoding="utf-8") as fp:
        d = yaml.safe_load(fp) or {}

    scales = d.get("scales")
    if scales:
        print(f"{f.relative_to(root)} sizes: {list(scales.keys())}")
