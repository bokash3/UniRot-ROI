import os, argparse, cv2, torch, json
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets import SegmentationDataset, list_images
from src.models.unet_seg import UNetSeg

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--out_masks_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out_masks_dir, exist_ok=True)

    files = list_images(args.images_dir)
    ds = SegmentationDataset(args.images_dir, None, files, is_train=False)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try common encoders
    for enc in ["resnet34", "mobilenet_v2"]:
        try:
            model = UNetSeg(encoder_name=enc).to(device)
            sd = torch.load(args.ckpt, map_location=device)["model"]
            model.load_state_dict(sd)
            break
        except Exception as e:
            last_err = e
            model = None
    if model is None:
        raise RuntimeError(f"Failed to load checkpoint. Last error: {last_err}")
    model.eval()

    with torch.no_grad():
        for imgs, _, names in tqdm(ld, desc="Infer seg"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i, name in enumerate(names):
                mask = (probs[i,0] > 0.5).astype("uint8") * 255
                outp = os.path.join(args.out_masks_dir, os.path.splitext(name)[0] + ".png")
                cv2.imwrite(outp, mask)

    print("Saved predicted masks to:", args.out_masks_dir)

if __name__ == "__main__":
    main()
