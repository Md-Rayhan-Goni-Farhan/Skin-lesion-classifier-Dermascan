"""
Downloads HAM10000 skin lesion dataset from Harvard Dataverse.
Resumes interrupted downloads. Run again if it fails.
"""

import os, sys, csv, random, shutil, zipfile, time, requests
from pathlib import Path

SEED      = 42
VAL_SPLIT = 0.2
DATA_DIR  = Path("data")
CACHE_DIR = Path("data/_cache")
random.seed(SEED)

FILES = [
    ("https://dataverse.harvard.edu/api/access/datafile/3172585", "HAM10000_images_part1.zip"),
    ("https://dataverse.harvard.edu/api/access/datafile/3172584", "HAM10000_images_part2.zip"),
    ("https://dataverse.harvard.edu/api/access/datafile/3172582", "HAM10000_metadata.csv"),
]

LABEL_MAP = {
    "mel":   "malignant",
    "bcc":   "malignant",
    "akiec": "malignant",
    "nv":    "benign",
    "bkl":   "benign",
    "df":    "benign",
    "vasc":  "benign",
}

CACHE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = CACHE_DIR / "images_all"
IMAGES_DIR.mkdir(exist_ok=True)
for split in ("train", "val"):
    for cls in ("benign", "malignant"):
        (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def download_file(url, dest):
    dest = Path(dest)
    tmp  = Path(str(dest) + ".part")

    # Already fully downloaded - skip immediately
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"  Already complete: {dest.name}")
        return

    # Delete stale .part if dest now exists (rename already happened)
    if dest.exists() and tmp.exists():
        tmp.unlink()
        print(f"  Already complete: {dest.name}")
        return

    existing = tmp.stat().st_size if tmp.exists() else 0
    if existing:
        print(f"  Resuming {dest.name} from {existing/1_048_576:.1f} MB...")
    else:
        print(f"  Downloading {dest.name}...")

    total = 0
    for attempt in range(999):
        existing = tmp.stat().st_size if tmp.exists() else 0
        headers  = {"User-Agent": "Mozilla/5.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            r = requests.get(url, stream=True, timeout=60, headers=headers)
            if r.status_code == 416:
                # Server says range not satisfiable = file complete
                if tmp.exists():
                    tmp.replace(dest)
                return
            r.raise_for_status()
            if not total:
                total = int(r.headers.get("content-length", 0))

            mode = "ab" if existing and r.status_code == 206 else "wb"
            if mode == "wb":
                existing = 0

            with open(tmp, mode) as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
                    existing += len(chunk)
                    if total:
                        pct = existing * 100 // total
                        sys.stdout.write(f"\r    {pct}%  {existing/1_048_576:.1f}/{total/1_048_576:.1f} MB")
                        sys.stdout.flush()
            print()

            # Rename only if dest doesn't exist yet
            if not dest.exists():
                tmp.rename(dest)
            elif tmp.exists():
                tmp.unlink()
            return

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\n  Dropped ({type(e).__name__}). Retry {attempt+1} in 5s...")
            time.sleep(5)

    raise RuntimeError(f"Failed after many attempts: {dest.name}")


def extract_zip(zip_path):
    done_flag = IMAGES_DIR / (Path(zip_path).stem + ".extracted")
    if done_flag.exists():
        print(f"  Already extracted: {Path(zip_path).name}")
        return
    print(f"  Extracting {Path(zip_path).name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        members = [m for m in z.namelist() if m.lower().endswith(".jpg")]
        for i, m in enumerate(members, 1):
            z.extract(m, IMAGES_DIR)
            if i % 200 == 0:
                sys.stdout.write(f"\r    {i}/{len(members)}")
                sys.stdout.flush()
    print()
    Path(done_flag).touch()


def main():
    print("=== Downloading HAM10000 Dataset ===")
    print("(Run again if connection drops - it resumes)\n")

    for url, filename in FILES:
        download_file(url, CACHE_DIR / filename)

    for _, filename in FILES:
        if filename.endswith(".zip"):
            extract_zip(CACHE_DIR / filename)

    all_imgs = {p.stem: p for p in IMAGES_DIR.rglob("*.jpg")}
    all_imgs.update({p.stem: p for p in IMAGES_DIR.rglob("*.JPG")})
    print(f"\nFound {len(all_imgs)} images.")

    csv_path = CACHE_DIR / "HAM10000_metadata.csv"
    records  = {"benign": [], "malignant": []}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            img_id = row.get("image_id", "").strip().strip('"')
            label  = LABEL_MAP.get(row.get("dx", "").strip().strip('"').lower())
            if label and img_id in all_imgs:
                records[label].append((img_id, all_imgs[img_id]))

    print(f"Labelled: {len(records['benign'])} benign, {len(records['malignant'])} malignant")

    min_count = min(len(records["benign"]), len(records["malignant"]))
    for k in records:
        random.shuffle(records[k])
        records[k] = records[k][:min_count]

    for cls, items in records.items():
        n_val = int(len(items) * VAL_SPLIT)
        for split_name, subset in [("val", items[:n_val]), ("train", items[n_val:])]:
            dest_dir = DATA_DIR / split_name / cls
            print(f"Copying {split_name}/{cls}: {len(subset)} images")
            for img_id, src in subset:
                shutil.copy2(src, dest_dir / f"{img_id}.jpg")

    print("\n✓ Dataset ready.")
    for split in ("train", "val"):
        for cls in ("benign", "malignant"):
            n = len(list((DATA_DIR / split / cls).glob("*.jpg")))
            print(f"  {split}/{cls}: {n} images")

if __name__ == "__main__":
    main()
