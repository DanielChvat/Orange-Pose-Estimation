import argparse
import os
import shutil


KEEP_OUT = {
    "frames",
    "gaussian_mask_votes",
    "gaussian_ownership",
    "gaussian_object_cutouts",
    "gaussian_object_tracks",
    "refined_object_gaussians",
    "masks",
    "colmap",
    "3dgs",
}

KEEP_VIS = {
    "gaussian_splat_overlay",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Remove stale experiment outputs and keep canonical project artifacts.")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--vis-dir", type=str, default="vis")
    parser.add_argument("--apply", action="store_true", help="Actually delete paths. Without this, only print what would be removed.")
    return parser.parse_args()


def clean_dir(root, keep, apply):
    if not os.path.isdir(root):
        return []
    removed = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if name in keep:
            continue
        removed.append(path)
        if apply:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    return removed


def main():
    args = parse_args()
    removed = clean_dir(args.out_dir, KEEP_OUT, args.apply)
    removed += clean_dir(args.vis_dir, KEEP_VIS, args.apply)
    action = "Removed" if args.apply else "Would remove"
    print(f"[INFO] {action} {len(removed)} paths")
    for path in removed:
        print(path)


if __name__ == "__main__":
    main()
