# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Benchmark a YOLO model formats for speed and accuracy.

Usage:
    from ultralytics.utils.benchmarks import ProfileModels, benchmark
    ProfileModels(['yolo11n.yaml', 'yolov8s.yaml']).run()
    benchmark(model='yolo11n.pt', imgsz=160)

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolo11n.pt
TorchScript             | `torchscript`             | yolo11n.torchscript
ONNX                    | `onnx`                    | yolo11n.onnx
OpenVINO                | `openvino`                | yolo11n_openvino_model/
TensorRT                | `engine`                  | yolo11n.engine
CoreML                  | `coreml`                  | yolo11n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolo11n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo11n.pb
TensorFlow Lite         | `tflite`                  | yolo11n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolo11n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolo11n_web_model/
PaddlePaddle            | `paddle`                  | yolo11n_paddle_model/
MNN                     | `mnn`                     | yolo11n.mnn
NCNN                    | `ncnn`                    | yolo11n_ncnn_model/
IMX                     | `imx`                     | yolo11n_imx_model/
RKNN                    | `rknn`                    | yolo11n_rknn_model/
ExecuTorch              | `executorch`              | yolo11n_executorch_model/
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch.cuda

from ultralytics import YOLO, YOLOWorld
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.engine.exporter import export_formats
from ultralytics.utils import (
    ARM64,
    ASSETS,
    ASSETS_URL,
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    TQDM,
    WEIGHTS_DIR,
    YAML,
)
from ultralytics.utils.checks import IS_PYTHON_3_13, check_imgsz, check_requirements, check_yolo, is_rockchip
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import get_cpu_info, select_device


def benchmark(
    model=WEIGHTS_DIR / "yolo11n.pt",
    data=None,
    imgsz=160,
    half=False,
    int8=False,
    device="cpu",
    verbose=False,
    eps=1e-3,
    format="",
    **kwargs,
):
    """Benchmark a YOLO model across different formats for speed and accuracy.

    Args:
        model (str | Path): Path to the model file or directory.
        data (str | None): Dataset to evaluate on, inherited from TASK2DATA if not passed.
        imgsz (int): Image size for the benchmark.
        half (bool): Use half-precision for the model if True.
        int8 (bool): Use int8-precision for the model if True.
        device (str): Device to run the benchmark on, either 'cpu' or 'cuda'.
        verbose (bool | float): If True or a float, assert benchmarks pass with given metric.
        eps (float): Epsilon value for divide by zero prevention.
        format (str): Export format for benchmarking. If not supplied all formats are benchmarked.
        **kwargs (Any): Additional keyword arguments for exporter.

    Returns:
        (polars.DataFrame): A polars DataFrame with benchmark results for each format, including file size, metric, and
            inference time.

    Examples:
        Benchmark a YOLO model with default settings:
        >>> from ultralytics.utils.benchmarks import benchmark
        >>> benchmark(model="yolo11n.pt", imgsz=640)
    """
    imgsz = check_imgsz(imgsz)
    assert imgsz[0] == imgsz[1] if isinstance(imgsz, list) else True, "benchmark() only supports square imgsz."

    import polars as pl  # scope for faster 'import ultralytics'

    pl.Config.set_tbl_cols(-1)  # Show all columns
    pl.Config.set_tbl_rows(-1)  # Show all rows
    pl.Config.set_tbl_width_chars(-1)  # No width limit
    pl.Config.set_tbl_hide_column_data_types(True)  # Hide data types
    pl.Config.set_tbl_hide_dataframe_shape(True)  # Hide shape info
    pl.Config.set_tbl_formatting("ASCII_BORDERS_ONLY_CONDENSED")

    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    is_end2end = getattr(model.model.model[-1], "end2end", False)
    data = data or TASK2DATA[model.task]  # task to dataset, i.e. coco8.yaml for task=detect
    key = TASK2METRIC[model.task]  # task to metric, i.e. metrics/mAP50-95(B) for task=detect

    y = []
    t0 = time.time()

    format_arg = format.lower()
    if format_arg:
        formats = frozenset(export_formats()["Argument"])
        assert format in formats, f"Expected format to be one of {formats}, but got '{format_arg}'."
    for name, format, suffix, cpu, gpu, _ in zip(*export_formats().values()):
        emoji, filename = "❌", None  # export defaults
        # try:
        if format_arg and format_arg != format:
            continue

        # Checks
        if format == "pb":
            assert model.task != "obb", "TensorFlow GraphDef not supported for OBB task"
        elif format == "edgetpu":
            assert LINUX and not ARM64, "Edge TPU export only supported on non-aarch64 Linux"
        elif format in {"coreml", "tfjs"}:
            assert MACOS or (LINUX and not ARM64), (
                "CoreML and TF.js export only supported on macOS and non-aarch64 Linux"
            )
        if format == "coreml":
            assert not IS_PYTHON_3_13, "CoreML not supported on Python 3.13"
        if format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            assert not isinstance(model, YOLOWorld), "YOLOWorldv2 TensorFlow exports not supported by onnx2tf yet"
            # assert not IS_PYTHON_MINIMUM_3_12, "TFLite exports not supported on Python>=3.12 yet"
        if format == "paddle":
            assert not isinstance(model, YOLOWorld), "YOLOWorldv2 Paddle exports not supported yet"
            assert model.task != "obb", "Paddle OBB bug https://github.com/PaddlePaddle/Paddle/issues/72024"
            assert not is_end2end, "End-to-end models not supported by PaddlePaddle yet"
            assert (LINUX and not IS_JETSON) or MACOS, "Windows and Jetson Paddle exports not supported yet"
        if format == "mnn":
            assert not isinstance(model, YOLOWorld), "YOLOWorldv2 MNN exports not supported yet"
        if format == "ncnn":
            assert not isinstance(model, YOLOWorld), "YOLOWorldv2 NCNN exports not supported yet"
        if format == "imx":
            assert not is_end2end
            assert not isinstance(model, YOLOWorld), "YOLOWorldv2 IMX exports not supported"
            assert model.task in {"detect", "classify", "pose"}, (
                "IMX export is only supported for detection, classification and pose estimation tasks"
            )
            assert "C2f" in model.__str__(), "IMX only supported for YOLOv8n and YOLO11n"
        if format == "rknn":
            assert not isinstance(model, YOLOWorld), "YOLOWorldv2 RKNN exports not supported yet"
            assert not is_end2end, "End-to-end models not supported by RKNN yet"
            assert LINUX, "RKNN only supported on Linux"
            assert is_rockchip(), "RKNN Inference only supported on Rockchip devices"
        if format == "executorch":
            assert not isinstance(model, YOLOWorld), "YOLOWorldv2 ExecuTorch exports not supported yet"
            assert not is_end2end, "End-to-end models not supported by ExecuTorch yet"
        if "cpu" in device.type:
            assert cpu, "inference not supported on CPU"
        if "cuda" in device.type:
            assert gpu, "inference not supported on GPU"

        # Export
        if format == "-":
            filename = model.pt_path or model.ckpt_path or model.model_name
            exported_model = model  # PyTorch format
        else:
            filename = model.export(
                imgsz=imgsz, format=format, half=half, int8=int8, data=data, device=device, verbose=False, **kwargs
            )
            exported_model = YOLO(filename, task=model.task)
            assert suffix in str(filename), "export failed"
        emoji = "❎"  # indicates export succeeded

        # Predict
        assert model.task != "pose" or format != "pb", "GraphDef Pose inference is not supported"
        assert model.task != "pose" or format != "executorch", "ExecuTorch Pose inference is not supported"
        assert format not in {"edgetpu", "tfjs"}, "inference not supported"
        assert format != "coreml" or platform.system() == "Darwin", "inference only supported on macOS>=10.13"
        if format == "ncnn":
            assert not is_end2end, "End-to-end torch.topk operation is not supported for NCNN prediction yet"
        exported_model.predict(ASSETS / "bus.jpg", imgsz=imgsz, device=device, half=half, verbose=False)

        # Validate
        results = exported_model.val(
            data=data,
            batch=1,
            imgsz=imgsz,
            plots=False,
            device=device,
            half=half,
            int8=int8,
            verbose=False,
            conf=0.001,  # all the pre-set benchmark mAP values are based on conf=0.001
        )
        metric, speed = results.results_dict[key], results.speed["inference"]
        fps = round(1000 / (speed + eps), 2)  # frames per second
        y.append([name, "✅", round(file_size(filename), 1), round(metric, 4), round(speed, 2), fps])
        # except Exception as e:
        #     if verbose:
        #         assert type(e) is AssertionError, f"Benchmark failure for {name}: {e}"
        #     LOGGER.error(f"Benchmark failure for {name}: {e}")
        #     y.append([name, emoji, round(file_size(filename), 1), None, None, None])  # mAP, t_inference

    # Print results
    check_yolo(device=device)  # print system info
    df = pl.DataFrame(y, schema=["Format", "Status❔", "Size (MB)", key, "Inference time (ms/im)", "FPS"], orient="row")
    df = df.with_row_index(" ", offset=1)  # add index info
    df_display = df.with_columns(pl.all().cast(pl.String).fill_null("-"))

    name = model.model_name
    dt = time.time() - t0
    legend = "Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed"
    s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({dt:.2f}s)\n{legend}\n{df_display}\n"
    LOGGER.info(s)
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics = df[key].to_numpy()  # values to compare to floor
        floor = verbose  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(x > floor for x in metrics if not np.isnan(x)), f"Benchmark failure: metric(s) < floor {floor}"

    return df_display


class RF100Benchmark:
    """Benchmark YOLO model performance across various formats for speed and accuracy.

    This class provides functionality to benchmark YOLO models on the RF100 dataset collection.

    Attributes:
        ds_names (list[str]): Names of datasets used for benchmarking.
        ds_cfg_list (list[Path]): List of paths to dataset configuration files.
        rf (Roboflow): Roboflow instance for accessing datasets.
        val_metrics (list[str]): Metrics used for validation.

    Methods:
        set_key: Set Roboflow API key for accessing datasets.
        parse_dataset: Parse dataset links and download datasets.
        fix_yaml: Fix train and validation paths in YAML files.
        evaluate: Evaluate model performance on validation results.
    """

    def __init__(self):
        """Initialize the RF100Benchmark class for benchmarking YOLO model performance across various formats."""
        self.ds_names = []
        self.ds_cfg_list = []
        self.rf = None
        self.val_metrics = ["class", "images", "targets", "precision", "recall", "map50", "map95"]

    def set_key(self, api_key: str):
        """Set Roboflow API key for processing.

        Args:
            api_key (str): The API key.

        Examples:
            Set the Roboflow API key for accessing datasets:
            >>> benchmark = RF100Benchmark()
            >>> benchmark.set_key("your_roboflow_api_key")
        """
        check_requirements("roboflow")
        from roboflow import Roboflow

        self.rf = Roboflow(api_key=api_key)

    def parse_dataset(self, ds_link_txt: str = "datasets_links.txt"):
        """Parse dataset links and download datasets.

        Args:
            ds_link_txt (str): Path to the file containing dataset links.

        Returns:
            ds_names (list[str]): List of dataset names.
            ds_cfg_list (list[Path]): List of paths to dataset configuration files.

        Examples:
            >>> benchmark = RF100Benchmark()
            >>> benchmark.set_key("api_key")
            >>> benchmark.parse_dataset("datasets_links.txt")
        """
        (shutil.rmtree("rf-100"), os.mkdir("rf-100")) if os.path.exists("rf-100") else os.mkdir("rf-100")
        os.chdir("rf-100")
        os.mkdir("ultralytics-benchmarks")
        safe_download(f"{ASSETS_URL}/datasets_links.txt")

        with open(ds_link_txt, encoding="utf-8") as file:
            for line in file:
                try:
                    _, _url, workspace, project, version = re.split("/+", line.strip())
                    self.ds_names.append(project)
                    proj_version = f"{project}-{version}"
                    if not Path(proj_version).exists():
                        self.rf.workspace(workspace).project(project).version(version).download("yolov8")
                    else:
                        LOGGER.info("Dataset already downloaded.")
                    self.ds_cfg_list.append(Path.cwd() / proj_version / "data.yaml")
                except Exception:
                    continue

        return self.ds_names, self.ds_cfg_list

    @staticmethod
    def fix_yaml(path: Path):
        """Fix the train and validation paths in a given YAML file."""
        yaml_data = YAML.load(path)
        yaml_data["train"] = "train/images"
        yaml_data["val"] = "valid/images"
        YAML.dump(yaml_data, path)

    def evaluate(self, yaml_path: str, val_log_file: str, eval_log_file: str, list_ind: int):
        """Evaluate model performance on validation results.

        Args:
            yaml_path (str): Path to the YAML configuration file.
            val_log_file (str): Path to the validation log file.
            eval_log_file (str): Path to the evaluation log file.
            list_ind (int): Index of the current dataset in the list.

        Returns:
            (float): The mean average precision (mAP) value for the evaluated model.

        Examples:
            Evaluate a model on a specific dataset
            >>> benchmark = RF100Benchmark()
            >>> benchmark.evaluate("path/to/data.yaml", "path/to/val_log.txt", "path/to/eval_log.txt", 0)
        """
        skip_symbols = ["🚀", "⚠️", "💡", "❌"]
        class_names = YAML.load(yaml_path)["names"]
        with open(val_log_file, encoding="utf-8") as f:
            lines = f.readlines()
            eval_lines = []
            for line in lines:
                if any(symbol in line for symbol in skip_symbols):
                    continue
                entries = line.split(" ")
                entries = list(filter(lambda val: val != "", entries))
                entries = [e.strip("\n") for e in entries]
                eval_lines.extend(
                    {
                        "class": entries[0],
                        "images": entries[1],
                        "targets": entries[2],
                        "precision": entries[3],
                        "recall": entries[4],
                        "map50": entries[5],
                        "map95": entries[6],
                    }
                    for e in entries
                    if e in class_names or (e == "all" and "(AP)" not in entries and "(AR)" not in entries)
                )
        map_val = 0.0
        if len(eval_lines) > 1:
            LOGGER.info("Multiple dicts found")
            for lst in eval_lines:
                if lst["class"] == "all":
                    map_val = lst["map50"]
        else:
            LOGGER.info("Single dict found")
            map_val = next(res["map50"] for res in eval_lines)

        with open(eval_log_file, "a", encoding="utf-8") as f:
            f.write(f"{self.ds_names[list_ind]}: {map_val}\n")

        return float(map_val)


class ProfileModels:
    """ProfileModels class for profiling different models on ONNX and TensorRT.

    This class profiles the performance of different models, returning results such as model speed and FLOPs.

    Attributes:
        paths (list[str]): Paths of the models to profile.
        num_timed_runs (int): Number of timed runs for the profiling.
        num_warmup_runs (int): Number of warmup runs before profiling.
        min_time (float): Minimum number of seconds to profile for.
        imgsz (int): Image size used in the models.
        half (bool): Flag to indicate whether to use FP16 half-precision for profiling.
        int8 (bool): Flag to indicate whether to use INT8 quantization for profiling.
        device (torch.device): Device used for profiling.

    Methods:
        run: Profile YOLO models for speed and accuracy across various formats.
        get_files: Get all relevant model files.
        get_onnx_model_info: Extract metadata from an ONNX model.
        iterative_sigma_clipping: Apply sigma clipping to remove outliers.
        profile_tensorrt_model: Profile a TensorRT model.
        profile_onnx_model: Profile an ONNX model.
        generate_table_row: Generate a table row with model metrics.
        generate_results_dict: Generate a dictionary of profiling results.
        print_table: Print a formatted table of results.

    Examples:
        Profile models and print results
        >>> from ultralytics.utils.benchmarks import ProfileModels
        >>> profiler = ProfileModels(["yolo11n.yaml", "yolov8s.yaml"], imgsz=640)
        >>> profiler.run()
    """

    def __init__(
        self,
        paths: list[str],
        num_timed_runs: int = 20,
        num_warmup_runs: int = 5,
        min_time: float = 10.0,
        imgsz: int = 640,
        half: bool = True,
        int8: bool = False,
        data: Optional[str] = None,
        task: str = 'detect',
        export_formats: Optional[list[str]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Initialize the ProfileModels class for profiling models.

        Args:
            paths (list[str]): List of paths of the models to be profiled.
            num_timed_runs (int): Number of timed runs for the profiling.
            num_warmup_runs (int): Number of warmup runs before the actual profiling starts.
            min_time (float): Minimum time in seconds for profiling a model.
            imgsz (int): Size of the image used during profiling.
            half (bool): Flag to indicate whether to use FP16 half-precision for TensorRT profiling.
            int8 (bool): Flag to indicate whether to use INT8 quantization for profiling.
            export_formats (Optional[list[str]]): List of formats to profile.
            device (torch.device | None): Device used for profiling. If None, it is determined automatically.

        Examples:
            Initialize and profile models
            >>> from ultralytics.utils.benchmarks import ProfileModels
            >>> profiler = ProfileModels(
                paths=["yolov8n.yaml", "yolov8s.yaml"],
                imgsz=640,
                half=True,
                int8=False,
                export_formats=["onnx", "engine"],
            )
            >>> profiler.run()
        """
        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.min_time = min_time
        self.imgsz = imgsz
        self.half = half
        self.int8 = int8
        self.export_formats = export_formats
        self.task = task
        self.data = data

        try:
            device = select_device(device)
        except ValueError:
            device = "cpu"
            LOGGER.warning(f"Invalid device {device}, using CPU instead.")
        self.device = device

    def run(self):
        """Profile YOLO models for speed and accuracy across various formats including ONNX and TensorRT.

        Returns:
            (list[dict]): List of dictionaries containing profiling results for each model.

        Examples:
            Profile models and print results
            >>> from ultralytics.utils.benchmarks import ProfileModels
            >>> profiler = ProfileModels(["yolo11n.yaml", "yolov8s.yaml"])
            >>> results = profiler.run()
        """
        files = self._get_files()
        print(files)
        exportable_formats = {".pt", ".yaml", ".yml"}

        table_rows, output = [], []
        for file in files:
            if file.suffix in exportable_formats:
                # Source model files: export to each format, then benchmark
                model = YOLO(str(file), task=self.task)
                model.fuse()  # to report correct params and GFLOPs in model.info()
                model_info = model.info()

                for export_format in self.export_formats:
                    exported_file = model.export(
                        format=export_format,
                        half=self.half,
                        int8=self.int8,
                        data=self.data,
                        imgsz=self.imgsz,
                        device=self.device,
                        simplify=True,
                        verbose=False,
                        batch=1,
                        dynamic=False,
                    )
                    t_export_format = self.profile_export_format(exported_file)
                    full_format = export_format + (" half" if self.half else " int8")
                    table_rows.append(self.generate_table_row(file.stem, full_format, t_export_format, model_info))
                    output.append(self.generate_results_dict(file.stem, full_format, t_export_format, model_info))
            else:
                # Pre-exported format files (.rknn, .onnx, .engine, etc.): benchmark directly
                t_export_format = self.profile_export_format(str(file))
                model_info = (0, 0, 0, 0)  # layers, params, gradients, flops not available
                table_rows.append(self.generate_table_row(file.stem[:10], '', t_export_format, model_info))
                output.append(self.generate_results_dict(file.stem[:10], '', t_export_format, model_info))

        self.print_table(table_rows)
        return output

    def profile_export_format(self, exported_file, eps: float = 1e-3):
        """Profile YOLO model performance, measuring average run time and standard deviation.

        Args:
            exported_file (str | tuple): Path to the exported model file, or tuple where first element is the path.
            eps (float): Small epsilon value to prevent division by zero.

        Returns:
            tuple: Three tuples containing (mean, std) for inference, preprocess, and postprocess times in ms.
        """
        # Handle tuple case (e.g., some exports return (path, metadata))
        if isinstance(exported_file, tuple):
            exported_file = exported_file[0]

        # Check for file or directory (RKNN exports return a folder path)
        if not exported_file or not (Path(exported_file).is_file() or Path(exported_file).is_dir()):
            LOGGER.warning(f"File {exported_file} not found.")
            return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)

        try:
            # Model and input
            model = YOLO(exported_file, task=self.task)
            input_data = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)  # use uint8 for Classify

            # Warmup runs
            elapsed = 0.0
            for _ in range(3):
                start_time = time.time()
                for _ in range(self.num_warmup_runs):
                    _ = model.predict(input_data, imgsz=self.imgsz, verbose=False)
                elapsed = time.time() - start_time

            # Compute number of runs as higher of min_time or num_timed_runs
            num_runs = max(round(self.min_time / (elapsed + eps) * self.num_warmup_runs), self.num_timed_runs * 50)

            # Timed runs
            run_times, preprocess_times, postprocess_times = [], [], []
            for _ in TQDM(range(num_runs), desc=str(exported_file)):
                results = model.predict(input_data, imgsz=self.imgsz, verbose=False)
                run_times.append(results[0].speed["inference"])
                preprocess_times.append(results[0].speed["preprocess"])
                postprocess_times.append(results[0].speed["postprocess"])

            run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)
            preprocess_times = self.iterative_sigma_clipping(np.array(preprocess_times), sigma=2, max_iters=3)
            postprocess_times = self.iterative_sigma_clipping(np.array(postprocess_times), sigma=2, max_iters=3)
            return (
                (np.mean(run_times), np.std(run_times)),
                (np.mean(preprocess_times), np.std(preprocess_times)),
                (np.mean(postprocess_times), np.std(postprocess_times)),
            )
        except Exception as e:
            LOGGER.warning(f"Failed to profile {exported_file}: {e}")
            return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)

    def _get_files(self):
        """Returns a list of paths for all relevant model files given by the user."""
        files = []
        for path in self.paths:
            path = Path(path)
            if path.suffix in {".pt", ".yaml", ".yml"}:  # add non-existing
                files.append(str(path))
            else:
                files.extend(glob.glob(str(path)))

        return [Path(file) for file in sorted(files)]

    @staticmethod
    def iterative_sigma_clipping(data: np.ndarray, sigma: float = 2, max_iters: int = 3):
        """Apply iterative sigma clipping to data to remove outliers.

        Args:
            data (np.ndarray): Input data array.
            sigma (float): Number of standard deviations to use for clipping.
            max_iters (int): Maximum number of iterations for the clipping process.

        Returns:
            (np.ndarray): Clipped data array with outliers removed.
        """
        data = np.array(data)
        for _ in range(max_iters):
            mean, std = np.mean(data), np.std(data)
            clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
            if len(clipped_data) == len(data):
                break
            data = clipped_data
        return data

    @staticmethod
    def check_dynamic(tensor_shape):
        """Check whether the tensor shape in the ONNX model is dynamic."""
        return not all(isinstance(dim, int) and dim >= 0 for dim in tensor_shape)

    def generate_table_row(
        self,
        model_name: str,
        format_name: str,
        t_format: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        model_info: tuple[float, float, float, float],
    ):
        """Generate a table row string with model performance metrics.

        Args:
            model_name (str): Name of the model.
            format_name (str): Name of the export format (e.g., 'onnx', 'engine', 'rknn').
            t_format (tuple): Three tuples of (mean, std) for inference, preprocess, postprocess times.
            model_info (tuple): Model information (layers, params, gradients, flops).

        Returns:
            (str): Formatted table row string with model metrics.
        """
        t_inference, t_preprocess, t_postprocess = t_format
        _layers, params, _gradients, flops = model_info
        speed_str = f"{t_inference[0]:.2f}±{t_inference[1]:.2f}"
        pre_str = f"{t_preprocess[0]:.2f}"
        post_str = f"{t_postprocess[0]:.2f}"
        return (
            f"| {model_name:15s} | {format_name:10s} | {self.imgsz:^9} | {speed_str:^17} | "
            f"{pre_str:^8} | {post_str:^9} | {params / 1e6:^10.1f} | {flops:^9.1f} |"
        )

    @staticmethod
    def generate_results_dict(
        model_name: str,
        format_name: str,
        t_format: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        model_info: tuple[float, float, float, float],
    ):
        """Generate a dictionary of profiling results.

        Args:
            model_name (str): Name of the model.
            format_name (str): Name of the export format (e.g., 'onnx', 'engine', 'rknn').
            t_format (tuple): Three tuples of (mean, std) for inference, preprocess, postprocess times.
            model_info (tuple): Model information (layers, params, gradients, flops).

        Returns:
            (dict): Dictionary containing profiling results.
        """
        t_inference, t_preprocess, t_postprocess = t_format
        _layers, params, _gradients, flops = model_info
        return {
            "model/name": model_name,
            "model/format": format_name,
            "model/parameters": params,
            "model/GFLOPs": round(flops, 3),
            f"model/speed_{format_name}(ms)": round(t_inference[0], 3),
            f"model/speed_{format_name}_std(ms)": round(t_inference[1], 3),
            f"model/preprocess_{format_name}(ms)": round(t_preprocess[0], 3),
            f"model/postprocess_{format_name}(ms)": round(t_postprocess[0], 3),
        }

    @staticmethod
    def print_table(table_rows: list[str]):
        """Print a formatted table of model profiling results.

        Args:
            table_rows (list[str]): List of formatted table row strings.
        """
        headers = [
            ("Model", 15),
            ("Format", 10),
            ("Size (px)", 9),
            ("Inference (ms)", 17),
            ("Pre (ms)", 8),
            ("Post (ms)", 9),
            ("Params (M)", 10),
            ("FLOPs (B)", 9),
        ]
        header = "|" + "|".join(f" {h:^{w}} " for h, w in headers) + "|"
        separator = "|" + "|".join("-" * (w + 2) for _, w in headers) + "|"
        total_width = 1 + sum(w + 2 for _, w in headers) + len(headers)
        border = "-" * total_width

        LOGGER.info(f"\n\n{border}")
        LOGGER.info(header)
        LOGGER.info(separator)
        for row in table_rows:
            LOGGER.info(row)
        LOGGER.info(border)
