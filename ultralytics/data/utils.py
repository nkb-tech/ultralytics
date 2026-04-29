# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import hashlib
import json
import os
import random
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    ROOT,
    SETTINGS_FILE,
    TQDM,
    clean_url,
    colorstr,
    emojis,
    is_dir_writeable,
    YAML,
)
from ultralytics.utils.checks import check_file, check_font, is_ascii
from ultralytics.utils.downloads import download, safe_download, unzip_file
from ultralytics.utils.ops import segments2boxes

HELP_URL = "See https://docs.ultralytics.com/datasets for dataset formatting guidance."
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"


def _int_keys(d: dict) -> dict:
    """Return a shallow copy of a YAML dict with integer keys where possible."""
    return {int(k): v for k, v in d.items()}


def _is_compact_multitask_hierarchy(data: dict) -> bool:
    """Detect the semantic-task YAML format before ``check_class_names`` flattens it."""
    names = data.get("names")
    child_parent_map = data.get("child_parent_map")
    if not isinstance(names, dict) or not isinstance(child_parent_map, dict):
        return False
    sample_name = next(iter(names.values()), None)
    sample_map = next(iter(child_parent_map.values()), None)
    return isinstance(sample_name, (list, tuple, dict)) and isinstance(sample_map, dict) and (
        "main_task" in data or any(isinstance(v, dict) and "connections" in v for v in sample_map.values())
    )


def _normalize_name_list(names, dataset) -> dict[int, str]:
    """Normalize one class-name list/dict to the repo's list[dict] class-name contract."""
    normalized = check_class_names(names)
    if len(normalized) != 1:
        raise SyntaxError(emojis(f"{dataset} nested class-name entries must describe exactly one class set."))
    return normalized[0]


def _normalize_compact_multitask_hierarchy(data: dict, dataset) -> None:
    """Normalize compact semantic-task hierarchy YAML into flat heads plus routing metadata."""
    semantic_names = _int_keys(data["names"])
    raw_maps = _int_keys(data["child_parent_map"])
    main_task = int(data.get("main_task", 0))
    main_level = data.get("main_level", None)
    main_level = None if main_level is None else int(main_level)
    if main_task not in semantic_names:
        raise SyntaxError(emojis(f"{dataset} main_task={main_task} is not present in names."))

    flat_names, flat_nc, flat_to_semantic = [], [], []
    semantic_to_flat, label_nc, child_parent_map = {}, [], {}
    hierarchy_parent_heads = []
    source_label_heads = []

    for task_id in sorted(semantic_names):
        semantic_to_flat[task_id] = {}
        label_nc.append(len(_normalize_name_list(semantic_names[task_id], dataset)))
        task_map = raw_maps.get(task_id)

        if task_map is None:
            # No hierarchy block means a flat auxiliary semantic task.
            task_names = _normalize_name_list(semantic_names[task_id], dataset)
            flat_idx = len(flat_names)
            semantic_to_flat[task_id][0] = flat_idx
            flat_names.append(task_names)
            flat_nc.append(len(task_names))
            flat_to_semantic.append({"task": task_id, "level": 0})
            hierarchy_parent_heads.append(-1)
            source_label_heads.append(flat_idx)
            continue

        task_map = _int_keys(task_map)
        deepest_level = max(task_map)
        for level in sorted(task_map):
            level_info = task_map[level]
            if not isinstance(level_info, dict) or "names" not in level_info or "connections" not in level_info:
                raise SyntaxError(
                    emojis(f"{dataset} child_parent_map[{task_id}][{level}] must contain 'connections' and 'names'.")
                )
            level_names = _normalize_name_list(level_info["names"], dataset)
            flat_idx = len(flat_names)
            semantic_to_flat[task_id][level] = flat_idx
            flat_names.append(level_names)
            flat_nc.append(len(level_names))
            flat_to_semantic.append({"task": task_id, "level": level})

            connections = level_info["connections"]
            if connections == -1:
                hierarchy_parent_heads.append(-1)
            else:
                connections = _int_keys(connections)
                if level - 1 not in semantic_to_flat[task_id]:
                    raise SyntaxError(emojis(f"{dataset} hierarchy level {level} has no previous parent level."))
                parent_head = semantic_to_flat[task_id][level - 1]
                hierarchy_parent_heads.append(parent_head)
                child_parent_map[flat_idx] = connections

        deepest_names = flat_names[semantic_to_flat[task_id][deepest_level]]
        declared_names = _normalize_name_list(semantic_names[task_id], dataset)
        if list(deepest_names.values()) != list(declared_names.values()):
            raise SyntaxError(
                emojis(f"{dataset} names[{task_id}] must match child_parent_map[{task_id}][{deepest_level}].names.")
            )
        source_label_heads.append(semantic_to_flat[task_id][deepest_level])

    if main_level is None:
        main_head = source_label_heads[sorted(semantic_names).index(main_task)]
    else:
        if main_level not in semantic_to_flat.get(main_task, {}):
            raise SyntaxError(emojis(f"{dataset} main_level={main_level} is not present in main_task={main_task}."))
        main_head = semantic_to_flat[main_task][main_level]
    compact_ignore = _int_keys(data.get("ignore_class", {})) if isinstance(data.get("ignore_class"), dict) else {}
    flat_ignore = {}
    for task_id, levels in compact_ignore.items():
        if not isinstance(levels, dict):
            continue
        for level, classes in _int_keys(levels).items():
            flat_idx = semantic_to_flat.get(int(task_id), {}).get(int(level))
            if flat_idx is not None:
                flat_ignore[flat_idx] = [int(c) for c in classes]

    data["names"] = flat_names
    data["nc"] = flat_nc
    data["child_parent_map"] = child_parent_map
    data["ignore_class"] = flat_ignore
    data["task_schema"] = {
        "format": "compact_multitask_hierarchy",
        "semantic_tasks": sorted(semantic_names),
        "semantic_nc": label_nc,
        "label_nc": label_nc,
        "semantic_to_flat": semantic_to_flat,
        "flat_to_semantic": flat_to_semantic,
        "source_label_heads": source_label_heads,
        "main_task": main_task,
        "main_level": main_level,
        "main_head": main_head,
        "hierarchy_parent_heads": hierarchy_parent_heads,
        "child_parent_map": child_parent_map,
        "ignore_class": flat_ignore,
    }


def _expand_compact_label_rows(lb: np.ndarray, task_schema: dict) -> np.ndarray:
    """Expand compact semantic-task labels to one class column per flat model head."""
    compact_cols = len(task_schema["label_nc"])
    flat_heads = len(task_schema["flat_to_semantic"])
    expanded = np.zeros((lb.shape[0], flat_heads), dtype=lb.dtype)
    hierarchy_parent_heads = task_schema["hierarchy_parent_heads"]
    child_parent_map = {int(k): _int_keys(v) for k, v in task_schema.get("child_parent_map", {}).items()}

    for label_col, source_head in enumerate(task_schema["source_label_heads"]):
        source_head = int(source_head)
        cls_values = lb[:, label_col].astype(np.int64, copy=False)
        expanded[:, source_head] = cls_values
        child_head = source_head
        child_cls = cls_values
        while int(hierarchy_parent_heads[child_head]) >= 0:
            parent_head = int(hierarchy_parent_heads[child_head])
            mapping = child_parent_map.get(child_head, {})
            try:
                parent_cls = np.array([mapping[int(c)] for c in child_cls], dtype=np.int64)
            except KeyError as e:
                raise ValueError(f"compact hierarchy label class {int(e.args[0])} has no parent mapping") from e
            expanded[:, parent_head] = parent_cls
            child_head, child_cls = parent_head, parent_cls

    return np.concatenate((expanded, lb[:, compact_cols : compact_cols + 4]), axis=1)


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
    return s


def verify_image(args, min_imgsz=25):
    """Verify one image."""
    (im_file, cls), prefix = args
    # Number (found, corrupt), message
    nf, nc, msg = 0, 0, ""
    try:
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] >= min_imgsz) & (shape[1] >= min_imgsz), f"image size {shape} <{min_imgsz} pixels"
        assert im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
    return (im_file, cls), nf, nc, msg


def verify_image_label(args, min_imgsz=9):
    """Verify one image-label pair."""
    if len(args) == 10:
        im_file, lb_file, prefix, keypoint, use_tags, n_tag_attrs, nkpt, ndim, single_cls, nc = args
        task_schema = None
    else:
        im_file, lb_file, prefix, keypoint, use_tags, n_tag_attrs, nkpt, ndim, single_cls, nc, task_schema = args
    # nc is a list with number of classes for each attribute (multi-head support)
    if not isinstance(nc, (list, tuple)):
        raise ValueError("'nc' must be a list specifying number of classes per attribute (multi-head labels)")
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    compact_schema = (
        task_schema if isinstance(task_schema, dict) and task_schema.get("format") == "compact_multitask_hierarchy" else None
    )
    label_nc = compact_schema["label_nc"] if compact_schema else nc
    nm, nf, ne, ncpt, msg, segments, keypoints, tags, nattrs = 0, 0, 0, 0, "", [], None, None, len(label_nc)
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > min_imgsz) & (shape[1] > min_imgsz), f"image size {shape} <{min_imgsz} pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                # Detect segment format: minimum 3 points per polygon (3*2 - 1 = 5)
                if any(len(x) > nattrs + 5 for x in lb) and (not keypoint):
                    classes = np.array([x[:nattrs] for x in lb], dtype=np.float32)
                    segments = [np.array(x[nattrs:], dtype=np.float32).reshape(-1, 2) for x in lb]  # polygon
                    lb = np.concatenate((classes, segments2boxes(segments)), 1)  # concat classes + xywh
                else:
                    lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    expected_cols = nattrs + 4 + nkpt * ndim
                    assert lb.shape[1] == expected_cols, (
                        f"labels require {expected_cols} columns each (got {lb.shape[1]}). "
                        f"Expected {nattrs} class attrs, 4 bbox, {nkpt*ndim} keypoint values"
                    )
                    points = lb[:, nattrs + 4:].reshape(-1, ndim)[:, :2]
                else:
                    # bbox points must NEVER include tags
                    if use_tags:
                        exp_a = nattrs + n_tag_attrs + 4          # tags after class attrs
                        exp_b = nattrs + 4 + n_tag_attrs          # tags after bbox (your val_mot_id case)

                        if lb.shape[1] == exp_b:
                            # [class_attrs..., x y w h, tag...]
                            points = lb[:, nattrs : nattrs + 4]
                            tags = lb[:, nattrs + 4 : nattrs + 4 + n_tag_attrs].astype(np.int64)
                            lb = lb[:, : nattrs + 4]  # keep only class attrs + bbox in lb
                        elif lb.shape[1] == exp_a:
                            # [class_attrs..., tag..., x y w h]
                            tags = lb[:, nattrs : nattrs + n_tag_attrs].astype(np.int64)
                            points = lb[:, nattrs + n_tag_attrs : nattrs + n_tag_attrs + 4]
                            # rebuild lb to class attrs + bbox
                            lb = np.concatenate([lb[:, :nattrs], points], axis=1).astype(np.float32)
                        else:
                            expected_cols = nattrs + 4
                            assert lb.shape[1] == expected_cols, (
                                f"labels require {expected_cols} columns (got {lb.shape[1]}). "
                                f"Expected {nattrs} class attrs + 4 bbox coords"
                            )
                            points = lb[:, nattrs : nattrs + 4]
                            tags = None
                    else:
                        expected_cols = nattrs + 4
                        assert lb.shape[1] == expected_cols, (
                            f"labels require {expected_cols} columns (got {lb.shape[1]}). "
                            f"Expected {nattrs} class attrs + 4 bbox coords"
                        )
                        points = lb[:, nattrs : nattrs + 4]
                # Coordinate points check with 1% tolerance
                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"negative class labels {lb[lb < -0.01]}"

                # All labels
                for i, nc_i in enumerate(label_nc):
                    # TODO make single cls work with multi-head labels
                    if single_cls and i == 0:
                        lb[:, i] = 0
                    max_cls = lb[:, i].max()
                    assert max_cls < nc_i, (
                        f"Label class {int(max_cls)} for attribute {i} exceeds dataset class count {nc_i}. "
                        f"Allowed range: 0-{nc_i - 1}"
                    )
                # remove duplicate rows
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}{im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, nattrs + 4 + (nkpt * ndim if keypoint else 0)), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, nattrs + 4 + (nkpt * ndim if keypoint else 0)), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, nattrs + 4 :].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, : nattrs + 4]
        if compact_schema:
            lb = _expand_compact_label_rows(lb, compact_schema)
        return im_file, lb, shape, segments, keypoints, tags, nm, nf, ne, ncpt, msg
    except Exception as e:
        ncpt = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, None, nm, nf, ne, ncpt, msg]


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int, optional): The color value to fill in the polygons on the mask. Defaults to 1.
        downsample_ratio (int, optional): Factor by which to downsample the mask. Defaults to 1.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Convert a list of polygons to a set of binary masks of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int): The color value to fill in the polygons on the masks.
        downsample_ratio (int, optional): Factor by which to downsample each mask. Defaults to 1.

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask.astype(masks.dtype))
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def find_dataset_yaml(path: Path) -> Path:
    """
    Find and return the YAML file associated with a Detect, Segment or Pose dataset.

    This function searches for a YAML file at the root level of the provided directory first, and if not found, it
    performs a recursive search. It prefers YAML files that have the same stem as the provided path. An AssertionError
    is raised if no YAML file is found or if multiple YAML files are found.

    Args:
        path (Path): The directory path to search for the YAML file.

    Returns:
        (Path): The path of the found YAML file.
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  # try root level first and then recursive
    assert files, f"No YAML file found in '{path.resolve()}'"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]  # prefer *.yaml files that match
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"
    return files[0]


def check_det_dataset(dataset, autodownload=True):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.

    Returns:
        (dict): Parsed dataset information and paths.
    """
    file = check_file(dataset)

    # Download (optional)
    extract_dir = ""
    if zipfile.is_zipfile(file) or is_tarfile(file):
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False

    # Read YAML
    data = YAML.load(file, append_filename=True)  # dictionary

    # Checks
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
                )
            LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
            data["val"] = data.pop("validation")  # replace 'validation' key with 'val' key
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if _is_compact_multitask_hierarchy(data):
        _normalize_compact_multitask_hierarchy(data, dataset)
    if "names" in data and "nc" in data and \
        (len(data["names"]) != len(data["nc"]) or \
        not all(len(names) == nci for names, nci in zip(data["names"], data["nc"]))):
        raise SyntaxError(emojis(f"{dataset} 'names' length {data['names']} and 'nc: {data['nc']}' must match."))
    if "names" in data:
        raw_names = data.get("names")
        if isinstance(raw_names, list) and raw_names and isinstance(raw_names[0], (list, tuple, dict)):
            # Multi-task format: list of lists, list of tuples, or list of dicts
            # check_class_names returns [normalized_dict], so unwrap with [0]
            data["names"] = [check_class_names(n)[0] for n in raw_names]
        elif isinstance(raw_names, (dict, list)):
            # Single-task format: dict or flat list of names - check_class_names returns a list
            data["names"] = check_class_names(raw_names)
        else:
            raise SyntaxError(emojis(f"{dataset} 'names' must be a list of lists, a list, or a dictionary."))
        data["nc"] = [len(names) for names in data["names"]]
    elif "nc" in data:
        nc = data["nc"]
        if isinstance(nc, list):
            data["names"] = [
                {i: f"class_{i}" for i in range(nci)}
                for nci in nc
            ]
        else:
            raise SyntaxError(emojis(f"{dataset} 'nc' must be a list."))

    # Resolve paths
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()

    # Set paths
    data["path"] = path  # download scripts
    for k in "train", "val", "test", "minival":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse YAML
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # dataset name with URL auth stripped
            m = f"\nDataset '{name}' images not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_FILE}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None  # success
            if s.startswith("http") and s.endswith(".zip"):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                r = os.system(s)
            else:  # python script
                exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in {0, None} else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}\n")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf")  # download fonts

    return data  # dictionary


def check_cls_dataset(dataset, split=""):
    """
    Checks a classification dataset such as Imagenet, now with .yaml support.
    
    This function first checks if `dataset` is a YAML file. If so, it parses the YAML similar to `check_det_dataset`.
    Otherwise, it treats `dataset` as a directory (or attempts download if it's a known dataset name or URL).
    
    Args:
        dataset (str | Path): The name or path of the dataset or a YAML file describing it.
        split (str, optional): The split of the dataset. Either 'val', 'test', or ''. Defaults to ''.

    Returns:
        (dict): A dictionary containing:
            - 'train' (list[Path]): List of directories/paths containing the training set of the dataset.
            - 'val' (list[Path]): List of directories/paths containing the validation set of the dataset.
            - 'test' (list[Path]): List of directories/paths containing the test set of the dataset (if any).
            - 'nc' (int): The number of classes in the dataset.
            - 'names' (dict): A dictionary of class names in the dataset.
    """
    # If dataset is a URL or archive, attempt download/unzip
    if str(dataset).startswith(("http:/", "https:/")) or Path(dataset).suffix in {".zip", ".tar", ".gz"}:
        file = check_file(dataset)
        dataset = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        dataset = Path(dataset).resolve()

    dataset = Path(dataset)

    # Check if dataset is a YAML file
    if dataset.suffix == ".yaml":
        # Load YAML
        data = YAML.load(dataset, append_filename=True)

        # Check required keys
        for k in ["train", "val"]:
            if k not in data and (k != "val" or "validation" not in data):
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
                )
            if k == "val" and "val" not in data and "validation" in data:
                LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
                data["val"] = data.pop("validation")

        if "names" not in data and "nc" not in data:
            raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required."))

        if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
            raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} does not match 'nc: {data['nc']}'."))

        # If only nc given, create generic names
        if "names" not in data:
            data["names"] = [f"class_{i}" for i in range(data["nc"])]
        else:
            data["nc"] = len(data["names"])

        data["names"] = check_class_names(data["names"])

        # Resolve dataset root path
        path = Path(data.get("path", "")).resolve() if "path" in data else dataset.parent
        if not path.is_absolute():
            path = (DATASETS_DIR / path).resolve()
        data["path"] = path

        # Convert train/val/test to absolute paths
        def make_paths(x):
            if not x:
                return []
            if isinstance(x, str):
                x = [x]
            return [str((path / p).resolve()) for p in x]

        train_paths = make_paths(data.get("train", []))
        val_paths = make_paths(data.get("val", []))
        test_paths = make_paths(data.get("test", []))

        # If split is specified, we can adjust sets accordingly
        # However, for classification, we usually rely on directories only.
        # split='val' or 'test' means we might just return val or test sets.
        # For now, we assume user wants entire dataset info, not filtered by split.
        # If split == 'val', we might just return val sets as primary sets, and so forth.
        
        # If nc/names provided by YAML, we don't need to scan. But we must ensure directories exist
        # If directories not found or empty, warn or raise error.
        # Check for existence of provided directories
        def check_dirs(dirs, name):
            checked = []
            for d in dirs:
                p = Path(d)
                if not p.exists():
                    raise FileNotFoundError(emojis(f"'{name}' directory {p} not found ❌"))
                checked.append(p)
            return checked

        train_dirs = check_dirs(train_paths, "train")
        val_dirs = check_dirs(val_paths, "val")
        test_dirs = check_dirs(test_paths, "test") if test_paths else []

        nc = data["nc"]
        names = {i: n for i, n in enumerate(data["names"].values())}

    else:
        # Original behavior: dataset is a directory
        data_dir = dataset if dataset.is_dir() else (DATASETS_DIR / dataset)
        data_dir = data_dir.resolve()
        if not data_dir.is_dir():
            LOGGER.warning(f"\nDataset not found ⚠️, missing path {data_dir}, attempting download...")
            t = time.time()
            if str(dataset) == "imagenet":
                subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
            else:
                url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{dataset.name}.zip"
                download(url, dir=data_dir.parent)
            s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
            LOGGER.info(s)

        train_dirs = [data_dir / "train"]
        val_dirs = [data_dir / "val"] if (data_dir / "val").exists() else [data_dir / "validation"] if (data_dir / "validation").exists() else []
        test_dirs = [data_dir / "test"] if (data_dir / "test").exists() else []

        # Infer nc and names from directory structure
        # We assume a standard classification folder structure:
        # train/class_0/*.jpg, train/class_1/*.jpg, ...
        train_dir = train_dirs[0] if train_dirs else None
        if train_dir and train_dir.exists():
            # number of classes
            sub_dirs = [x for x in train_dir.glob("*") if x.is_dir()]
            nc = len(sub_dirs)
            names = dict(enumerate(sorted([x.name for x in sub_dirs])))
        else:
            # No train_dir found, can't infer classes
            raise FileNotFoundError(emojis(f"{dataset} 'train:' directory not found ❌"))

    # Print dataset info for train, val, test
    def log_dataset_info(name, dirs):
        if not dirs:
            LOGGER.info(f"{colorstr(f'{name}:')} None")
            return
        for d in dirs:
            prefix = f'{colorstr(f"{name}:")} {d}...'
            if not d.exists():
                LOGGER.warning(f"{prefix} directory does not exist ⚠️")
                continue
            files = [path for path in d.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
            nf = len(files)
            # Count how many unique directories we have (i.e., classes)
            nd = len({file.parent for file in files})
            if name == "train" and nf == 0:
                raise FileNotFoundError(emojis(f"{d} '{name}:' no training images found ❌ "))
            elif nf == 0:
                LOGGER.warning(f"{prefix} found {nf} images in {nd} classes: WARNING ⚠️ no images found")
            elif nd != nc:
                LOGGER.warning(f"{prefix} found {nf} images in {nd} classes: ERROR ❌ requires {nc} classes, not {nd}")
            else:
                LOGGER.info(f"{prefix} found {nf} images in {nd} classes ✅ ")

    log_dataset_info("train", train_dirs)
    log_dataset_info("val", val_dirs)
    log_dataset_info("test", test_dirs)

    return {
        "train": train_dirs,
        "val": val_dirs,
        "test": test_dirs,
        "nc": nc,
        "names": names
    }


class HUBDatasetStats:
    """
    A class for generating HUB dataset JSON and `-hub` dataset directory.

    Args:
        path (str): Path to data.yaml or data.zip (with data.yaml inside data.zip). Default is 'coco8.yaml'.
        task (str): Dataset task. Options are 'detect', 'segment', 'pose', 'classify'. Default is 'detect'.
        autodownload (bool): Attempt to download dataset if not found locally. Default is False.

    Example:
        Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets
            i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.
        ```python
        from ultralytics.data.utils import HUBDatasetStats

        stats = HUBDatasetStats("path/to/coco8.zip", task="detect")  # detect dataset
        stats = HUBDatasetStats("path/to/coco8-seg.zip", task="segment")  # segment dataset
        stats = HUBDatasetStats("path/to/coco8-pose.zip", task="pose")  # pose dataset
        stats = HUBDatasetStats("path/to/dota8.zip", task="obb")  # OBB dataset
        stats = HUBDatasetStats("path/to/imagenet10.zip", task="classify")  # classification dataset

        stats.get_json(save=True)
        stats.process_images()
        ```
    """

    def __init__(self, path="coco8.yaml", task="detect", autodownload=False):
        """Initialize class."""
        path = Path(path).resolve()
        LOGGER.info(f"Starting HUB dataset checks for {path}....")

        self.task = task  # detect, segment, pose, classify, obb
        if self.task == "classify":
            unzip_dir = unzip_file(path)
            data = check_cls_dataset(unzip_dir)
            data["path"] = unzip_dir
        else:  # detect, segment, pose, obb
            _, data_dir, yaml_path = self._unzip(Path(path))
            try:
                # Load YAML with checks
                data = YAML.load(yaml_path)
                data["path"] = ""  # strip path since YAML should be in dataset root for all HUB datasets
                YAML.save(yaml_path, data)
                data = check_det_dataset(yaml_path, autodownload)  # dict
                data["path"] = data_dir  # YAML path should be set to '' (relative) or parent (absolute)
            except Exception as e:
                raise Exception("error/HUB/dataset_stats/init") from e

        self.hub_dir = Path(f'{data["path"]}-hub')
        self.im_dir = self.hub_dir / "images"
        self.stats = {"nc": len(data["names"]), "names": list(data["names"].values())}  # statistics dictionary
        self.data = data

    @staticmethod
    def _unzip(path):
        """Unzip data.zip."""
        if not str(path).endswith(".zip"):  # path is data.yaml
            return False, None, path
        unzip_dir = unzip_file(path, path=path.parent)
        assert unzip_dir.is_dir(), (
            f"Error unzipping {path}, {unzip_dir} not found. " f"path/to/abc.zip MUST unzip to path/to/abc/"
        )
        return True, str(unzip_dir), find_dataset_yaml(unzip_dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f):
        """Saves a compressed image for HUB previews."""
        compress_one_image(f, self.im_dir / Path(f).name)  # save to dataset-hub

    def get_json(self, save=False, verbose=False):
        """Return dataset JSON for Ultralytics HUB."""

        def _round(labels):
            """Update labels to integer class and 4 decimal place floats."""
            if self.task == "detect":
                coordinates = labels["bboxes"]
            elif self.task in {"segment", "obb"}:  # Segment and OBB use segments. OBB segments are normalized xyxyxyxy
                coordinates = [x.flatten() for x in labels["segments"]]
            elif self.task == "pose":
                n, nk, nd = labels["keypoints"].shape
                coordinates = np.concatenate((labels["bboxes"], labels["keypoints"].reshape(n, nk * nd)), 1)
            else:
                raise ValueError(f"Undefined dataset task={self.task}.")
            zipped = zip(labels["cls"], coordinates)
            return [[int(c[0]), *(round(float(x), 4) for x in points)] for c, points in zipped]

        for split in "train", "val", "test":
            self.stats[split] = None  # predefine
            path = self.data.get(split)

            # Check split
            if path is None:  # no split
                continue
            files = [f for f in Path(path).rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]  # image files in split
            if not files:  # no images
                continue

            # Get dataset statistics
            if self.task == "classify":
                from torchvision.datasets import ImageFolder

                dataset = ImageFolder(self.data[split])

                x = np.zeros(len(dataset.classes)).astype(int)
                for im in dataset.imgs:
                    x[im[1]] += 1

                self.stats[split] = {
                    "instance_stats": {"total": len(dataset), "per_class": x.tolist()},
                    "image_stats": {"total": len(dataset), "unlabelled": 0, "per_class": x.tolist()},
                    "labels": [{Path(k).name: v} for k, v in dataset.imgs],
                }
            else:
                from ultralytics.data import YOLODataset

                dataset = YOLODataset(img_path=self.data[split], data=self.data, task=self.task)
                x = np.array(
                    [
                        np.bincount(label["cls"].astype(int).flatten(), minlength=self.data["nc"])
                        for label in TQDM(dataset.labels, total=len(dataset), desc="Statistics")
                    ]
                )  # shape(128x80)
                self.stats[split] = {
                    "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
                    "image_stats": {
                        "total": len(dataset),
                        "unlabelled": int(np.all(x == 0, 1).sum()),
                        "per_class": (x > 0).sum(0).tolist(),
                    },
                    "labels": [{Path(k).name: _round(v)} for k, v in zip(dataset.im_files, dataset.labels)],
                }

        # Save, print and return
        if save:
            self.hub_dir.mkdir(parents=True, exist_ok=True)  # makes dataset-hub/
            stats_path = self.hub_dir / "stats.json"
            LOGGER.info(f"Saving {stats_path.resolve()}...")
            with open(stats_path, "w") as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):
        """Compress images for Ultralytics HUB."""
        from ultralytics.data import YOLODataset  # ClassificationDataset

        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes dataset-hub/images/
        for split in "train", "val", "test":
            if self.data.get(split) is None:
                continue
            dataset = YOLODataset(img_path=self.data[split], data=self.data)
            with ThreadPool(NUM_THREADS) as pool:
                for _ in TQDM(pool.imap(self._hub_ops, dataset.im_files), total=len(dataset), desc=f"{split} images"):
                    pass
        LOGGER.info(f"Done. All images saved to {self.im_dir}")
        return self.im_dir


def compress_one_image(f, f_new=None, max_dim=1920, quality=50):
    """
    Compresses a single image file to reduced size while preserving its aspect ratio and quality using either the Python
    Imaging Library (PIL) or OpenCV library. If the input image is smaller than the maximum dimension, it will not be
    resized.

    Args:
        f (str): The path to the input image file.
        f_new (str, optional): The path to the output image file. If not specified, the input file will be overwritten.
        max_dim (int, optional): The maximum dimension (width or height) of the output image. Default is 1920 pixels.
        quality (int, optional): The image compression quality as a percentage. Default is 50%.

    Example:
        ```python
        from pathlib import Path
        from ultralytics.data.utils import compress_one_image

        for f in Path("path/to/dataset").rglob("*.jpg"):
            compress_one_image(f)
        ```
    """
    try:  # use PIL
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, "JPEG", quality=quality, optimize=True)  # save
    except Exception as e:  # use OpenCV
        LOGGER.info(f"WARNING ⚠️ HUB ops PIL failure {f}: {e}")
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)  # ratio
        if r < 1.0:  # image too large
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)


def autosplit(path=DATASETS_DIR / "coco8/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
    """
    Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt files.

    Args:
        path (Path, optional): Path to images directory. Defaults to DATASETS_DIR / 'coco8/images'.
        weights (list | tuple, optional): Train, validation, and test split fractions. Defaults to (0.9, 0.1, 0.0).
        annotated_only (bool, optional): If True, only images with an associated txt file are used. Defaults to False.

    Example:
        ```python
        from ultralytics.data.utils import autosplit

        autosplit()
        ```
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], "a") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # add image to txt file


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x, version):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = version  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")