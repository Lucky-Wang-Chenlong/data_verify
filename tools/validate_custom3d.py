"""Validate a custom Omni3D-format dataset and render GT boxes for inspection.

Checks (errors stop training from working; warnings are suspicious):
  * schema: required image / annotation / category fields, unique ids
  * geometry: dims > 0, R_cam orthonormal with det = +1, center in front of
    the camera, bbox3D_cam == R_cam @ corners(dims) + center_cam (Omni3D
    corner order), bbox2D_proj == projection of bbox3D_cam
  * depth: file exists, decodes, shape matches the image, and the GT depth at
    each box's projected center agrees with the box (catches mm/m scale bugs
    and camera-frame mistakes)
  * filtering preview: how many boxes the train-time filters will ignore

Usage (or just scripts/validate.sh):
    python $DV/tools/validate_custom3d.py --root data/custom/toy --name Toy_train \
        --vis 20 --out $DV_OUT/vis/Toy_train
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import cv2
import numpy as np

IMAGE_FIELDS = ("id", "file_path", "depth_path", "width", "height", "K")
ANN_FIELDS = (
    "id", "image_id", "category_id", "category_name", "center_cam",
    "dimensions", "R_cam", "bbox3D_cam", "bbox2D_proj", "bbox2D_tight",
    "bbox2D_trunc", "behind_camera", "valid3D", "truncation", "visibility",
    "lidar_pts", "segmentation_pts", "depth_error",
)
# Box edges for the Omni3D corner order (0-3 back face, 4-7 front face).
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def omni3d_corners(center, dims, R) -> np.ndarray:
    """8 corners in camera frame; dims = [W, H, L], object x=L, y=H, z=W."""
    w, h, l = dims
    x = np.array([-l, l, l, -l, -l, l, l, -l]) / 2
    y = np.array([-h, -h, h, h, -h, -h, h, h]) / 2
    z = np.array([-w, -w, -w, -w, w, w, w, w]) / 2
    return (np.asarray(R) @ np.stack([x, y, z])).T + np.asarray(center)


def project(K: np.ndarray, pts: np.ndarray) -> np.ndarray:
    uvw = (K @ pts.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def load_depth(path: str, depth_scale: float) -> np.ndarray | None:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return d.astype(np.float32) / depth_scale


class Report:
    def __init__(self) -> None:
        self.errors: collections.Counter = collections.Counter()
        self.warnings: collections.Counter = collections.Counter()
        self.examples: dict[str, str] = {}

    def err(self, key: str, example: str) -> None:
        self.errors[key] += 1
        self.examples.setdefault(key, example)

    def warn(self, key: str, example: str) -> None:
        self.warnings[key] += 1
        self.examples.setdefault(key, example)

    def print(self) -> None:
        for title, counter in (("ERRORS", self.errors),
                               ("WARNINGS", self.warnings)):
            print(f"\n{title}: {sum(counter.values())}")
            for k, n in counter.most_common():
                print(f"  [{n:6d}] {k}   e.g. {self.examples[k]}")


def draw(img: np.ndarray, K: np.ndarray, anns: list[dict]) -> np.ndarray:
    out = img.copy()
    for a in anns:
        corners = np.asarray(a["bbox3D_cam"])
        if (corners[:, 2] <= 0.05).any():
            continue
        uv = project(K, corners).astype(int)
        color = (0, 255, 0) if a["valid3D"] else (0, 0, 255)
        for i, j in EDGES:
            cv2.line(out, tuple(uv[i]), tuple(uv[j]), color, 2)
        # Front face (+z of the object = +W) cross, shows yaw.
        cv2.line(out, tuple(uv[4]), tuple(uv[6]), (255, 128, 0), 1)
        cv2.line(out, tuple(uv[5]), tuple(uv[7]), (255, 128, 0), 1)
        x1, y1, x2, y2 = a["bbox2D_tight"] if a["bbox2D_tight"][0] != -1 \
            else a["bbox2D_trunc"]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      (255, 255, 0), 1)
        cv2.putText(out, a["category_name"], (int(x1), max(int(y1) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--depth_scale", type=float, default=1000.0)
    p.add_argument("--depth_tol", type=float, default=0.15,
                   help="relative tolerance for depth-vs-box agreement")
    p.add_argument("--vis", type=int, default=0, help="#images to render")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(os.path.join(args.root, "annotations", f"{args.name}.json")) as f:
        data = json.load(f)
    r = Report()

    # ---------------- schema ----------------
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    if not cats:
        r.err("no categories", "categories list empty")
    images = {}
    for im in data["images"]:
        missing = [k for k in IMAGE_FIELDS if k not in im]
        if missing:
            r.err(f"image missing {missing}", f"image id={im.get('id')}")
            continue
        if im["id"] in images:
            r.err("duplicate image id", str(im["id"]))
        images[im["id"]] = im
    anns_by_img = collections.defaultdict(list)
    ann_ids = set()
    for a in data["annotations"]:
        missing = [k for k in ANN_FIELDS if k not in a]
        if missing:
            r.err(f"annotation missing {missing}", f"ann id={a.get('id')}")
            continue
        if a["id"] in ann_ids:
            r.err("duplicate annotation id", str(a["id"]))
        ann_ids.add(a["id"])
        if a["image_id"] not in images:
            r.err("annotation -> unknown image_id", str(a["image_id"]))
            continue
        if cats.get(a["category_id"]) != a["category_name"]:
            r.err("category_id/category_name mismatch with categories",
                  f"ann {a['id']}: {a['category_id']}/{a['category_name']}")
        anns_by_img[a["image_id"]].append(a)

    # ---------------- geometry ----------------
    n_filtered = 0
    for img_id, anns in anns_by_img.items():
        im = images[img_id]
        K = np.asarray(im["K"], dtype=np.float64)
        if K.shape != (3, 3) or K[2, 2] != 1 or K[0, 0] <= 0:
            r.err("bad K (need 3x3, fx>0, K[2,2]=1)", f"image {img_id}")
            continue
        for a in anns:
            ex = f"image {img_id} ann {a['id']}"
            if not a["valid3D"]:
                # Placeholder 3D fields are allowed; the box is ignored.
                r.warn("valid3D=False (box ignored)", ex)
                n_filtered += 1
                continue
            dims = np.asarray(a["dimensions"], dtype=np.float64)
            R = np.asarray(a["R_cam"], dtype=np.float64)
            c = np.asarray(a["center_cam"], dtype=np.float64)
            if (dims <= 0).any():
                r.err("non-positive dimensions", ex)
                continue
            orth_err = np.abs(R @ R.T - np.eye(3)).max()
            if orth_err > 1e-2 or np.linalg.det(R) < 0:
                r.err("R_cam not a rotation (orthonormal, det=+1)", ex)
                continue
            if orth_err > 1e-4:
                r.warn("R_cam slightly non-orthonormal (1e-4..1e-2)", ex)
            if c[2] <= 0 and not a["behind_camera"]:
                r.err("center_cam z<=0 but behind_camera=False", ex)
            corners = np.asarray(a["bbox3D_cam"], dtype=np.float64)
            if corners.shape != (8, 3):
                r.err("bbox3D_cam must be 8x3", ex)
                continue
            if np.abs(omni3d_corners(c, dims, R) - corners).max() > 1e-3:
                r.err("bbox3D_cam != R_cam @ corners([W,H,L]) + center "
                      "(dims order or corner order wrong)", ex)
            if (corners[:, 2] > 0).all():
                uv = project(K, corners)
                proj = [*uv.min(0), *uv.max(0)]
                if np.abs(np.array(proj) - a["bbox2D_proj"]).max() > 2.0:
                    r.err("bbox2D_proj != projection of bbox3D_cam (K or "
                          "camera frame wrong)", ex)
            for key in ("bbox2D_tight", "bbox2D_trunc"):
                b = a[key]
                if b[0] != -1 and (b[2] <= b[0] or b[3] <= b[1]):
                    r.err(f"{key} not xyxy with x2>x1, y2>y1", ex)
            # Mirror the train-time filter (truncation<0.5, visibility>0.1).
            if a["behind_camera"] or (
                a["truncation"] >= 0.5) or (
                0 <= a["visibility"] <= 0.1) or a["lidar_pts"] == 0 or \
                    a["segmentation_pts"] == 0 or a["depth_error"] > 0.5:
                n_filtered += 1

    # ---------------- depth ----------------
    for img_id, im in images.items():
        path = os.path.join(args.root, im["depth_path"])
        if not os.path.exists(path):
            r.err("depth file missing (sample would be DROPPED)", path)
            continue
        if not os.path.exists(os.path.join(args.root, im["file_path"])):
            r.err("image file missing", im["file_path"])
        depth = load_depth(path, args.depth_scale)
        if depth is None:
            r.err("depth file unreadable", path)
            continue
        if depth.shape[:2] != (im["height"], im["width"]):
            r.err("depth shape != image (height, width)",
                  f"{path}: {depth.shape} vs {(im['height'], im['width'])}")
            continue
        if path.endswith(".png") and cv2.imread(
                path, cv2.IMREAD_UNCHANGED).dtype != np.uint16:
            r.err("depth PNG must be 16-bit", path)
        K = np.asarray(im["K"], dtype=np.float64)
        for a in anns_by_img.get(img_id, []):
            c = np.asarray(a["center_cam"], dtype=np.float64)
            if not a["valid3D"] or c[2] <= 0:
                continue
            u, v = project(K, c[None])[0]
            if not (0 <= u < im["width"] and 0 <= v < im["height"]):
                continue
            # The surface seen at the box center lies between the box's
            # nearest point and its far side: z in [zc - ext, zc + ext].
            ui, vi = int(u), int(v)
            patch = depth[max(vi - 2, 0):vi + 3, max(ui - 2, 0):ui + 3]
            d = np.median(patch[patch > 0]) if (patch > 0).any() else 0
            if d == 0:
                r.warn("no depth at box center", f"image {img_id}")
                continue
            ext = 0.5 * np.linalg.norm(a["dimensions"])
            lo, hi = c[2] - ext, c[2] + ext
            if not (lo * (1 - args.depth_tol) <= d <= hi * (1 + args.depth_tol)):
                r.warn("GT depth at box center inconsistent with box "
                       "(depth scale? occlusion?)",
                       f"image {img_id} ann {a['id']}: depth {d:.2f}m, "
                       f"box z {c[2]:.2f}±{ext:.2f}m")

    # ---------------- summary ----------------
    n_ann = sum(len(v) for v in anns_by_img.values())
    print(f"{args.name}: {len(images)} images, {n_ann} boxes, "
          f"{len(images) - len(anns_by_img)} images without boxes")
    print(f"categories: {cats}")
    print("boxes per category:",
          dict(collections.Counter(a["category_name"]
                                   for v in anns_by_img.values() for a in v)))
    print(f"boxes ignored by train-time filters: {n_filtered}/{n_ann}")
    r.print()

    # ---------------- visualization ----------------
    if args.vis > 0:
        out = args.out or os.path.join("vis", args.name)
        os.makedirs(out, exist_ok=True)
        for img_id in sorted(anns_by_img)[: args.vis]:
            im = images[img_id]
            img = cv2.imread(os.path.join(args.root, im["file_path"]))
            if img is None:
                continue
            vis = draw(img, np.asarray(im["K"]), anns_by_img[img_id])
            cv2.imwrite(os.path.join(out, f"{img_id:06d}.jpg"), vis)
        print(f"\nrendered {min(args.vis, len(anns_by_img))} images to {out}"
              " (green=3D box, cyan=2D box, orange cross=+W face)")

    raise SystemExit(1 if r.errors else 0)


if __name__ == "__main__":
    main()
