import os, json, argparse
from src.datasets import list_images, build_subject_split, load_angles

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--angles_json", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seeds", type=int, nargs="+", default=[42,2025,1337])
    args = p.parse_args()

    files = list_images(args.images_dir)
    # ensure angles exist for all files we care about
    meta = load_angles(args.angles_json)
    files = [f for f in files if f in meta]

    out = {}
    for s in args.seeds:
        tr, va, te = build_subject_split(files, args.train_ratio, args.val_ratio, args.test_ratio, seed=s)
        out[str(s)] = {"train": tr, "val": va, "test": te}

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved split to:", args.out_path)

if __name__ == "__main__":
    main()
