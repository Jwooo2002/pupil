dOeh# Comparison Report: Local Project vs. Upstream Pupil Labs

**Date:** 2026-02-13
**Upstream Repository:** [pupil-labs/pupil](https://github.com/pupil-labs/pupil)
**Local Project:** Custom Fork (PyTorch & DVS Integration)

## 1. Summary Table

| File / Module | Original Functionality | Your Modified Functionality | Impact |
| :--- | :--- | :--- | :--- |
| `detector_2d_plugin.py` | Wrapper for C++ geometric 2D pupil detection (contours, edge fitting). | **Replaced/Augmented** with PyTorch-based Deep Learning model (`DenseNet`). | **High**: Shifts core detection algorithm from CV to DL. Requires GPU/CUDA. |
| `dvs_detector_plugin.py` | *Does not exist* (Standard Pupil supports frame-based cameras). | **New Plugin**: Implements support for Dynamic Vision Sensors (DVS). | **High**: Enables neuromorphic event-based eye tracking using `dv_processing` and `tonic`. |
| `requirements.txt` | Standard scientific stack (`numpy`, `opencv`, `pyglui`, `zeromq`). | **Missing Dependencies**: Code requires `torch`, `torchvision`, `tonic`, `dv_processing` but they are not listed. | **Critical**: `pip install -r requirements.txt` will result in a broken environment. |
| `main.py` | Standard entry point. | Added Environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`). | **Low**: Optimization for multi-core processing (likely for PyTorch). |
| `eye.py` | Orchestrates the eye process and loads default plugins. | Modified to accommodate the custom detector initialization. | **Medium**: altered startup sequence for eye processes. |

## 2. Code Breakdown & Core Logic Deviations

### A. 2D Pupil Detection (`pupil_src/shared_modules/pupil_detector_plugins/detector_2d_plugin.py`)

**Original Upstream:**
The original plugin initializes a C++ object `Detector2D`. It relies on parameter tuning (thresholds, min/max radius) exposed via the UI.

**Your Modification:**
You have overridden the `__init__` method to load a PyTorch model. This suggests you are bypassing or augmenting the C++ detector with a learned model.

```python
# Your Code
def __init__(self, g_pool=None, properties=None, detector_2d: Detector2D = None):
    super().__init__(g_pool=g_pool)
    self.detector_2d = detector_2d or Detector2D(properties or {})
    
    # CUSTOM MODIFICATION: Deep Learning Initialization
    model_name = "densenet"
    model_path = "./best_model.pkl"
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = torch.device(device_str)
    # ... model loading logic ...
```

### B. DVS (Event Camera) Support (`pupil_src/shared_modules/pupil_detector_plugins/dvs_detector_plugin.py`)

**Original Upstream:**
No native support for Event Cameras.

**Your Modification:**
You created a completely new plugin class `DVSDetectorPlugin`. This integrates specific hardware SDKs (`dv_processing`) and neuromorphic learning libraries (`tonic`).

```python
# New File
class DVSDetectorPlugin(PupilDetectorPlugin):
    def __init__(self, g_pool, **config):
        # ...
        # Hardware Integration
        self.capture = CameraCapture() 
        self.slicer  = EventStreamSlicer()
        
        # Deep Learning for Events
        self.net = Model(args).cuda().eval()
```

## 3. Dependency & Environment Analysis

**Critical Gap Identified:**
Your code has introduced significant new dependencies that are **not reflected** in `requirements.txt`.

*   **Used in Code:** `torch`, `torchvision`, `tonic`, `dv_processing`, `PIL` (Pillow).
*   **Present in `requirements.txt`:** None of the above.

**Recommendation:**
You must update `requirements.txt` or create a `requirements-custom.txt` to ensure reproducibility.

## 4. Architectural Shifts

*   **Hybrid Architecture**: You are moving away from a purely CPU-based, lightweight C++ pipeline to a **GPU-heavy, Python-based Deep Learning pipeline**. This significantly increases the hardware requirements (CUDA-capable GPU recommended).
*   **Plugin System Usage**: You have correctly utilized the Pupil Labs plugin system (`PupilDetectorPlugin`) to extend functionality, which is the architectural "happy path." However, replacing the default `2d` detector's internals is invasive compared to registering a parallel detector.
*   **Threading**: The addition of `OMP_NUM_THREADS=4` in `main.py` indicates manual tuning for parallel processing, likely to prevent PyTorch or NumPy from monopolizing CPU cores and starving the real-time UI/Capture threads.

## 5. Comparison Summary

Your local project is a **specialized fork** of Pupil Capture designed for **next-generation eye tracking research**. 

1.  **AI-First Approach**: It replaces traditional computer vision with Deep Learning (DenseNet) for pupil detection.
2.  **Neuromorphic Hardware**: It introduces support for DVS cameras, a feature not present in the standard commercial version.
3.  **Prototype State**: The discrepancy in `requirements.txt` and the presence of "test" scripts (`pupil_detectors_test.py`, `detector_2d_plugin_cpu.py`) suggest this is an active research prototype rather than a polished distribution.
