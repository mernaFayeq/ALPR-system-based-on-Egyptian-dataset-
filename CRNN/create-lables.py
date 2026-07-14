import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent

image_dir = ROOT / "dataset" / "images"
output_csv = ROOT / "dataset" / "labels.csv"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not image_dir.exists():
    raise FileNotFoundError(f"Image folder not found: {image_dir}")

images = sorted([
    p.name for p in image_dir.iterdir()
    if p.suffix.lower() in image_extensions
])

output_csv.parent.mkdir(parents=True, exist_ok=True)

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "city",
        "plate_code",
        "plate_number",
        "ocr_label",
        "full_label",
        "quality",
        "split"
    ])

    for filename in images:
        writer.writerow([
            filename,
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ])

print(f"Created: {output_csv}")
print(f"Images found: {len(images)}")