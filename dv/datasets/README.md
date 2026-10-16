# Custom training data format

English | [中文](README_zh.md)

`scripts/train.sh --custom DIR` expects the dataset directory in the format below. It is Omni3D-style COCO JSON, plus a depth file path on every image. The loading code is in `custom3d.py` (the dataset class) and `config.py` (filter thresholds and augmentation).

When the data is ready, validate it first and fix every ERROR:

```bash
scripts/validate.sh --custom DIR      # GT box renders go to outputs/data_verify/vis/<split>/
```

For a worked example, see `data/custom/toy/`. `tools/make_toy_custom.py` builds it from the chairs in SUNRGBD_val.

---

## 1. Directory layout

```
DIR/                              # the directory name becomes part of the run name
├── annotations/
│   ├── <Prefix>_train.json       # training set (with --overfit, also used for evaluation)
│   └── <Prefix>_test.json        # test set (reported as AP_<Prefix>_test)
├── images/...                    # any layout; referenced by file_path in the JSON
└── depth/...                     # any layout; referenced by depth_path in the JSON
```

- **Split names**: `<Prefix>` is a name you choose, e.g. `MyCat`. By default the scripts pick the one `*_train.json` and the one `*_test.json` under `annotations/`. If there are several, pass the file stem (without `.json`) with `--train-split` / `--test-split`.
- **No Omni3D names**: don't use an Omni3D subset name such as `SUNRGBD` or `ARKitScenes` as the prefix. Otherwise the evaluation results get mixed with the standard test set.
- **Paths**: every path is relative to `DIR`.
- **Created during training**: `DIR/cache/<split>.pkl` holds the parsed annotations. **Delete this cache after you edit a JSON**, or training keeps reading the old annotations.
- **HDF5 (optional)**: if reading many small files is slow, run `python -m vis4d.data.io.to_hdf5 -p DIR/images` (and the same for `depth/`). Once `DIR/images.hdf5` exists, it is read from HDF5 automatically. The JSON paths stay the same.

## 2. Coordinate and unit conventions

| Item | Convention |
|---|---|
| Camera frame | OpenCV: x right, y down, z forward. Units are meters. |
| `K` | 3×3 intrinsics of the image **as stored**. If you resized or cropped the image, use the intrinsics after that step. |
| `dimensions` | **`[W, H, L]`** in meters (width, height, length) |
| `R_cam` | 3×3 rotation that maps object coordinates to camera coordinates. It must be orthonormal with det = +1. **Object frame: x along L, y along H, z along W.** |
| `center_cam` | 3D box center in the camera frame |
| `bbox3D_cam` | 8 corners = `R_cam @ c + center_cam`, where `c` = ½ × [(−L,−H,−W), (L,−H,−W), (L,H,−W), (−L,H,−W), (−L,−H,W), (L,−H,W), (L,H,W), (−L,H,W)] |
| 2D boxes | `[x1, y1, x2, y2]` in pixels; `[-1,-1,-1,-1]` when unavailable |
| Depth | Same height × width as the image. Either a 16-bit PNG (value = depth in meters × `--depth-scale`, default 1000, i.e. millimeters) or a float32 `.npy` in meters. 0 means invalid. Values above 20 m are dropped. |

The three most common mistakes are the order of `dimensions`, the mapping between object axes and dimensions, and the depth unit. The validator checks all three:
- it recomputes the corners from the convention and compares them with `bbox3D_cam`;
- it projects the corners into the image and compares them with `bbox2D_proj`;
- it reads the GT depth at each box center and checks that it agrees with the box depth.

Training uses `canonical_rotation=True`. The model normalizes boxes so that W ≤ L and yaw ∈ [0, π), so you don't need to resolve symmetric ambiguity yourself. But `R_cam`, the dimensions and the corners must agree with each other.

## 3. JSON fields

```jsonc
{
  "info": {"name": "MyCat_train"},
  "categories": [
    {"id": 0, "name": "my category"},   // the text prompt is this exact name
    {"id": 1, "name": "table"}          // optional: a category with no boxes acts as a negative prompt
  ],
  "images": [{
    "id": 0,                            // unique integer
    "file_path": "images/000000.jpg",
    "depth_path": "depth/000000.png",   // images without depth are dropped in training
    "width": 1280, "height": 720,
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  }],
  "annotations": [{
    "id": 0, "image_id": 0,
    "category_id": 0, "category_name": "my category",
    "center_cam": [x, y, z],
    "dimensions": [W, H, L],
    "R_cam": [[...], [...], [...]],
    "bbox3D_cam": [[x, y, z], ...8],
    "bbox2D_proj":  [x1, y1, x2, y2],   // bounding box of the projected bbox3D_cam (not clipped)
    "bbox2D_trunc": [x1, y1, x2, y2],   // bbox2D_proj clipped to the image
    "bbox2D_tight": [x1, y1, x2, y2],   // tight box of the visible mask, or -1s
    "behind_camera": false,
    "valid3D": true,                    // false = box is ignored (fields may hold placeholders)
    "truncation": 0.0,                  // fraction of the projected box outside the image, -1 if unknown
    "visibility": 1.0,                  // visible fraction (occlusion), -1 if unknown
    "lidar_pts": -1, "segmentation_pts": -1, "depth_error": -1   // Omni3D legacy fields; keep -1
  }]
}
```

The 2D box used for training is the first valid one in the order `bbox2D_trunc` → `bbox2D_proj` → `bbox2D_tight`.

## 4. Boxes that are ignored automatically

| | Training | Test and evaluation (Omni3D defaults) |
|---|---|---|
| truncation | ≥ 0.5 | ≥ 0.33 |
| visibility | ≤ 0.1 | ≤ 0.33 |
| 2D box height | no limit | < 6.25% of the image height |
| Ignored in both | `behind_camera`, `valid3D=false`, dimensions ≤ 0 | |

Ignored boxes are never regression targets. Detections on them are not counted as false positives either. In training, an image with no boxes left after filtering is dropped entirely.

At startup the training log prints the instance count per category (`Distribution of instances`). Use it to confirm that your category was loaded.

## 5. About depth

**Provide GT depth for every image.** There is no need to hold some images back without depth. The depth branch already drops its depth input at random during training: 70% of samples use no depth at all (monocular), 20% use patch-masked depth and 10% use full depth. So the model also learns the no-depth case.

The training loader drops any image without a `depth_path`.
