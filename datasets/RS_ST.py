import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import datasets.transform as transform
from torchvision.transforms import functional as F

DATASET_CONFIGS = {
    "SECOND": {
        "colormap": [[255,255,255], [0,128,0], [128,128,128], [0,255,0], [0,0,255], [128,0,0], [255,0,0]],
        "classes": ['unchanged', 'low vegetation', 'ground', 'tree', 'water', 'building', 'sports field'],
        "mean_A": np.array([113.40, 114.08, 116.45]),
        "std_A":  np.array([48.30,  46.27,  48.14]),
        "mean_B": np.array([111.07, 114.04, 118.18]),
        "std_B":  np.array([49.41,  47.01,  47.94]),
    },
    "Landsat": {
        "colormap": [[255,255,255], [0,155,0], [255,165,0], [230,30,100], [0,170,240]],
        "classes": ['unchanged', 'farmland', 'desert', 'building', 'water'],
        "mean_A": np.array([140.68, 137.93, 136.20]),
        "std_A":  np.array([81.46, 83.01, 83.70]),
        "mean_B": np.array([137.29, 136.14, 134.70]),
        "std_B":  np.array([84.65, 84.85, 85.80]),
    },
    "JL1H": {
        "colormap": [[255,255,255], [0,128,0], [0,0,128], [128,0,0], [0,128,128], [128,128,0]],
        "classes": ['unchanged', 'farm', 'road', 'tree', 'building', 'other'],
        "mean_A": np.array([113.40, 114.08, 116.45]),
        "std_A":  np.array([48.30,  46.27,  48.14]),
        "mean_B": np.array([111.07, 114.04, 118.18]),
        "std_B":  np.array([49.41,  47.01,  47.94]),
    },
    "HRSCD": {
        "colormap": [[255,255,255], [200,0,0], [255,200,0], [0,128,0], [0,200,200], [0,0,200]],
        "classes": ['unchanged', 'artificial surfaces', 'agricultural areas', 'forests', 'wetlands', 'water'],
        "mean_A": np.array([110.16, 112.09, 86.64]),
        "std_A":  np.array([54.89, 40.93, 38.67]),
        "mean_B": np.array([105.16, 119.17, 96.89]),
        "std_B":  np.array([50.14, 40.14, 34.96]),
    },
}

ST_COLORMAP = DATASET_CONFIGS["SECOND"]["colormap"]
ST_CLASSES = DATASET_CONFIGS["SECOND"]["classes"]
MEAN_A = DATASET_CONFIGS["SECOND"]["mean_A"]
STD_A  = DATASET_CONFIGS["SECOND"]["std_A"]
MEAN_B = DATASET_CONFIGS["SECOND"]["mean_B"]
STD_B  = DATASET_CONFIGS["SECOND"]["std_B"]
_LABEL_WARNED = False


def set_dataset_config(dataset_name):
    global ST_COLORMAP, ST_CLASSES, MEAN_A, STD_A, MEAN_B, STD_B
    cfg = DATASET_CONFIGS[dataset_name]
    ST_COLORMAP = cfg["colormap"]
    ST_CLASSES = cfg["classes"]
    MEAN_A = cfg["mean_A"]
    STD_A = cfg["std_A"]
    MEAN_B = cfg["mean_B"]
    STD_B = cfg["std_B"]
    print(f"[RS_ST] Dataset config set to '{dataset_name}': {len(ST_CLASSES)} classes")

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def Color2Index(label):
    """
    Convert color-coded label maps into class-index maps.
    Supports:
    - HxW single channel index mask
    - HxWx3 RGB color mask using ST_COLORMAP
    """
    global _LABEL_WARNED
    label = np.asarray(label)

    if label.ndim == 2:
        return label.astype(np.uint8)

    if label.ndim != 3 or label.shape[2] != 3:
        raise ValueError(f"Unsupported label shape: {label.shape}")

    colormap = np.asarray(ST_COLORMAP, dtype=np.uint8)
    index = np.zeros(label.shape[:2], dtype=np.uint8)
    matched = np.zeros(label.shape[:2], dtype=bool)

    for cls_idx, color in enumerate(colormap):
        mask = np.all(label == color, axis=-1)
        index[mask] = cls_idx
        matched |= mask

    if not np.all(matched) and not _LABEL_WARNED:
        _LABEL_WARNED = True
        unknown_pixels = int((~matched).sum())
        print(
            "[RS_ST] Warning: found pixels with colors not in ST_COLORMAP; "
            f"they are mapped to class 0. count={unknown_pixels}"
        )

    return index

def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time == 'A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im

def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs


def _pick_existing_dir(root, candidates):
    for name in candidates:
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"None of directories exists under {root}: {candidates}")


def _detect_ext(directory, default=".png"):
    """Return the most common image extension in *directory*."""
    exts = {}
    for f in os.listdir(directory):
        _, ext = os.path.splitext(f)
        ext = ext.lower()
        if ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
            exts[ext] = exts.get(ext, 0) + 1
    if not exts:
        return default
    return max(exts, key=exts.get)


class Data(data.Dataset):
    def __init__(self, datapath, mode, augmentation=False, num_classes=None):
        self.datapath = datapath
        self.mode = mode
        self.augmentation = augmentation
        self.num_classes = num_classes if num_classes else len(ST_COLORMAP)

        self.A = _pick_existing_dir(datapath, ["A", "im1"])
        self.B = _pick_existing_dir(datapath, ["B", "im2"])
        self.labels_A = _pick_existing_dir(datapath, ["label1", "labelA", "label1_gray"])
        self.labels_B = _pick_existing_dir(datapath, ["label2", "labelB", "label2_gray"])

        self.img_ext = _detect_ext(self.A)
        self.lbl_ext = _detect_ext(self.labels_A)

        self.list_img = self.get_mask_name(datapath)

    def get_mask_name(self, datapath):
        images_list_file = os.path.join(datapath, 'list', self.mode + ".txt")
        if os.path.isfile(images_list_file):
            with open(images_list_file, "r") as f:
                return f.readlines()

        IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
        names = [os.path.splitext(x)[0] for x in os.listdir(self.A)
                 if os.path.splitext(x)[1].lower() in IMG_EXTS]
        names = sorted(names)
        if len(names) == 0:
            raise FileNotFoundError(
                f"No list file found at {images_list_file}, and no image files found in {self.A}."
            )
        print(f"[RS_ST] list file not found for mode '{self.mode}', inferred {len(names)} IDs from {self.A}")
        return [n + "\n" for n in names]

    def __getitem__(self, idx):
        imgname = self.list_img[idx].strip('\n')
        imgname = os.path.splitext(imgname)[0]

        img_A = io.imread(os.path.join(self.A, imgname + self.img_ext))
        img_B = io.imread(os.path.join(self.B, imgname + self.img_ext))
        label_A = io.imread(os.path.join(self.labels_A, imgname + self.lbl_ext))
        label_B = io.imread(os.path.join(self.labels_B, imgname + self.lbl_ext))
        label_A = Color2Index(label_A)
        label_B = Color2Index(label_B)

        if self.augmentation:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_MCD(img_A, img_B, label_A, label_B)

        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')

        change_label = np.zeros_like(label_A, dtype=np.float32)
        change_label[(label_A != 0) | (label_B != 0)] = 1.0

        N = self.num_classes
        trans_label = label_A.astype(np.int64) * N + label_B.astype(np.int64)

        return (F.to_tensor(img_A), F.to_tensor(img_B),
                torch.from_numpy(label_A.copy()), torch.from_numpy(label_B.copy()),
                torch.from_numpy(change_label), torch.from_numpy(trans_label),
                imgname)

    def __len__(self):
        return len(self.list_img)


