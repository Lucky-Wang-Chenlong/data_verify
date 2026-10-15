# 自定义训练数据格式

`scripts/train.sh --custom DIR` 读取的数据集目录必须是下面的格式。整体是 Omni3D 风格的 COCO JSON，每张图额外带一个 depth 文件的路径。读取代码见 `custom3d.py`（数据集类）和 `config.py`（过滤阈值、数据增强）。

写好之后先跑一遍校验，有 ERROR 必须全部修掉：

```bash
scripts/validate.sh --custom DIR      # 渲染的 GT 框图在 outputs/data_verify/vis/<split>/
```

参考示例：`data/custom/toy/`，由 `tools/make_toy_custom.py` 从 SUNRGBD_val 里抽 chair 生成。

---

## 1. 目录结构

```
DIR/                              # 目录名会作为 run 名的一部分
├── annotations/
│   ├── <Prefix>_train.json       # 训练集（--overfit 时也用来评估）
│   └── <Prefix>_test.json        # 测试集（结果里显示为 AP_<Prefix>_test）
├── images/...                    # 任意子目录结构，由 JSON 里的 file_path 指定
└── depth/...                     # 任意子目录结构，由 JSON 里的 depth_path 指定
```

- **split 名**：`<Prefix>` 是你自己起的名字，例如 `MyCat`。默认会自动找 `annotations/` 下唯一的 `*_train.json` 和 `*_test.json`；有多个时，用 `--train-split` / `--test-split` 指定文件名（不带 `.json`）。
- **不能和 Omni3D 重名**：前缀不要用 `SUNRGBD`、`ARKitScenes` 这类 Omni3D 子集名，否则评估结果会和标准测试集混在一起。
- **路径**：所有路径都相对 `DIR`。
- **训练时会生成**：`DIR/cache/<split>.pkl` 是解析好的标注缓存。**改了 JSON 之后要删掉这个缓存**，否则会继续读旧标注。
- **改用 HDF5（可选）**：读大量小文件慢的时候，可以先运行 `python -m vis4d.data.io.to_hdf5 -p DIR/images`（`depth/` 同理）。生成 `DIR/images.hdf5` 后会自动改用 HDF5 读取，JSON 里的路径不用改。

## 2. 坐标与单位约定

| 项 | 约定 |
|---|---|
| 相机坐标系 | OpenCV：x 向右，y 向下，z 向前；单位米 |
| `K` | 3×3 内参，对应**实际存储的图像**（如果做过 resize 或 crop，要用处理之后的内参） |
| `dimensions` | **`[W, H, L]`**，单位米（宽、高、长） |
| `R_cam` | 3×3 旋转矩阵，把物体坐标变换到相机坐标。必须是正交矩阵且 det = +1。**物体坐标系：x 轴沿 L，y 轴沿 H，z 轴沿 W** |
| `center_cam` | 3D 框中心在相机坐标系下的位置 |
| `bbox3D_cam` | 8 个角点 = `R_cam @ c + center_cam`，其中 `c` = ½ × [(−L,−H,−W), (L,−H,−W), (L,H,−W), (−L,H,−W), (−L,−H,W), (L,−H,W), (L,H,W), (−L,H,W)] |
| 2D 框 | `[x1, y1, x2, y2]`，单位像素；没有时填 `[-1,-1,-1,-1]` |
| depth | 和图像同样的高 × 宽。可以是 16-bit PNG（值 = 深度米数 × `--depth-scale`，默认 1000，即毫米），也可以是 float32 的 `.npy`（单位米）。0 表示无效；超过 20 m 的值会被丢掉 |

最容易出错的三处是：`dimensions` 的顺序、物体坐标轴和尺寸的对应关系、depth 的单位。这三处校验器都会检查。它会按约定重新计算角点，并和 `bbox3D_cam` 对比；把角点投影到图像上，和 `bbox2D_proj` 对比；再读取框中心位置的 GT depth，看和框的深度是否一致。

训练时使用 `canonical_rotation=True`：模型会把框规范化成 W ≤ L、yaw ∈ [0, π)，对称造成的歧义不用你处理。但 `R_cam`、尺寸和角点三者之间必须自洽。

## 3. JSON 字段

```jsonc
{
  "info": {"name": "MyCat_train"},
  "categories": [
    {"id": 0, "name": "my category"},   // text prompt 就是这个名字，原样使用
    {"id": 1, "name": "table"}          // 可选：没有任何框的类别会被当作负样本 prompt
  ],
  "images": [{
    "id": 0,                            // 整数，不能重复
    "file_path": "images/000000.jpg",
    "depth_path": "depth/000000.png",   // 没有 depth 的图在训练时会被丢弃
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
    "bbox2D_proj":  [x1, y1, x2, y2],   // bbox3D_cam 投影到图像后的外接框（不裁剪）
    "bbox2D_trunc": [x1, y1, x2, y2],   // bbox2D_proj 裁剪到图像范围内
    "bbox2D_tight": [x1, y1, x2, y2],   // 可见 mask 的紧框；没有就填 -1
    "behind_camera": false,
    "valid3D": true,                    // false 表示这个框被忽略（字段可以填占位值）
    "truncation": 0.0,                  // 投影框落在图像外的比例；未知填 -1
    "visibility": 1.0,                  // 可见比例（遮挡）；未知填 -1
    "lidar_pts": -1, "segmentation_pts": -1, "depth_error": -1   // Omni3D 遗留字段，填 -1 即可
  }]
}
```

训练用的 2D 框按 `bbox2D_trunc` → `bbox2D_proj` → `bbox2D_tight` 的顺序，取第一个有效的。

## 4. 会被自动忽略的框

| | 训练 | 测试和评估（Omni3D 默认标准） |
|---|---|---|
| truncation | ≥ 0.5 | ≥ 0.33 |
| visibility | ≤ 0.1 | ≤ 0.33 |
| 2D 框高度 | 不限 | < 图像高度的 6.25% |
| 以下情况两边都忽略 | `behind_camera`、`valid3D=false`、尺寸 ≤ 0 | |

被忽略的框不会作为回归目标，但仍会抑制在它上面的误检，不会被当成错误检测。训练时，过滤后一个框都不剩的图会被整张丢掉。

训练日志开头会打印每个类别的实例数（`Distribution of instances`），可以用来确认你的类别确实被读进来了。

## 5. 关于 depth

**每张图都应该给 GT depth**，不需要刻意留一部分图不给。depth 分支训练时本身就会随机丢弃 depth 输入：70% 的样本完全不用 depth（单目），20% 用 patch 遮挡过的 depth，10% 用完整 depth。所以模型同时会学到没有 depth 时的情况。

没有 `depth_path` 的图会被训练 loader 直接丢掉。
