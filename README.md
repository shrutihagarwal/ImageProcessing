# 🖼️ Image Processing: From Scratch to OpenCV & PIL

A comprehensive implementation of classical image processing algorithms, built both **from mathematical first principles** and using industry-standard libraries (OpenCV, PIL). This project demonstrates deep understanding of the mathematics behind spatial-domain filtering, edge detection, and feature extraction.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Implementation Philosophy](#implementation-philosophy)
- [Project Structure](#project-structure)
- [Technologies & Libraries](#technologies--libraries)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Key Learnings](#key-learnings)

---

## Overview

This project systematically explores spatial-domain image processing techniques used in computer vision. Each algorithm is implemented in **two ways**:

1. **From Scratch** — using only NumPy and raw pixel manipulation to expose the mathematical core
2. **Using Libraries** — with OpenCV or PIL to validate results and demonstrate practical usage

This dual implementation approach builds intuition for *why* these algorithms work, not just *how* to call them.

---

## Algorithms Implemented

### 🔍 Feature Detection
| Algorithm | Description | Implementation |
|-----------|-------------|----------------|
| **Image Segmentation** | Line detection using directional kernels (horizontal, vertical, diagonal at 45°) | From scratch + OpenCV |
| **Canny Edge Detection** | Multi-stage edge detection via gradient magnitude and non-maximum suppression | OpenCV |
| **Harris Corner Detection** | Identifies corners using second-moment matrix eigenvalue analysis | From scratch + OpenCV |
| **Sobel Operator** | Gradient-based edge detection in x and y directions | From scratch + OpenCV |
| **Laplacian Filter** | Second-order derivative filter for edge enhancement (4 kernel variants) | From scratch + OpenCV |

### 🌫️ Smoothing & Noise Reduction
| Algorithm | Description | Implementation |
|-----------|-------------|----------------|
| **Gaussian Filter** | Weighted averaging with 3×3, 5×5, and 7×7 kernels | From scratch + OpenCV + PIL |
| **Median Filter** | Non-linear noise reduction (salt-and-pepper noise) | From scratch + OpenCV |
| **Mean / Average Filter** | Linear smoothing via box filtering | From scratch + OpenCV |
| **Weighted Average Filter** | Gaussian-weighted convolution for better smoothing | From scratch + OpenCV |
| **Bilateral Filter** | Edge-preserving smoothing with spatial and range Gaussian | OpenCV |

### 🎛️ Image Enhancement
| Algorithm | Description | Implementation |
|-----------|-------------|----------------|
| **Unsharp Masking** | Sharpening by subtracting a blurred version from the original | From scratch + PIL |
| **Minimum Filter** | Erosion-like morphological operation (neighbourhood minimum) | From scratch + PIL |
| **Maximum Filter** | Dilation-like morphological operation (neighbourhood maximum) | From scratch + PIL |
| **Region-based Gaussian Blur** | Selective blurring of image sub-regions | PIL |

---

## Implementation Philosophy

The core insight behind this project: **every spatial-domain filter is a convolution** (or related operation). Understanding this transforms a collection of algorithms into a unified framework.

For example, the **from-scratch Weighted Average Filter** explicitly:
1. Constructs the kernel matrix manually
2. Normalizes by the sum of weights
3. Slides the kernel over every pixel using nested loops
4. Computes the convolution output value per pixel

This makes the operation fully transparent, before confirming it with `cv2.filter2D`.

```python
# Example: Sobel operator from scratch
sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
v_feature = cv2.filter2D(img, -1, kernel=sobelx)
h_feature = cv2.filter2D(img, -1, kernel=sobely)
combined = v_feature + h_feature
```

---

## Project Structure

```
ImageProcessing/
├── Image Processing .py    # Main script with all implementations (1463 lines)
├── img1.png                # Sample test image 1
├── img3.jpg                # Sample test image 2
├── img4.jpg                # Sample test image 3
├── img5.jpg                # Sample test image 4
├── img6.jpg                # Sample test image 5 (primary test image)
└── README.md
```

---

## Technologies & Libraries

| Library | Usage |
|---------|-------|
| `numpy` | Kernel construction, pixel-level manipulation, from-scratch implementations |
| `opencv-python` (cv2) | Validation of scratch implementations, canny, harris, bilateral |
| `Pillow` (PIL) | Unsharp masking, min/max filters, Gaussian blur via PIL |
| `matplotlib` | Visual comparison of original vs. processed images |

---

## How to Run

### Prerequisites
```bash
pip install numpy opencv-python Pillow matplotlib
```

### Run the Script
```bash
python "Image Processing .py"
```

The script is structured as sections — each section demonstrates one algorithm with side-by-side visualizations (original vs. result).

> **Note:** The sample images (`img1.png`, `img3.jpg`–`img6.jpg`) must be in the same directory.

---

## Results Summary

Each algorithm is visually compared against the original image. Key observations:

- **Gaussian vs. Median**: Gaussian blur softens noise uniformly; Median filter better preserves edges while removing salt-and-pepper noise.
- **Sobel vs. Laplacian**: Sobel captures directional gradients; Laplacian is isotropic but sensitive to noise.
- **Harris Corner**: Successfully marks corner pixels in red, confirming the response is strongest at high curvature regions.
- **Bilateral Filter**: Most effective at edge-preserving smoothing — texture is retained while noise is suppressed.

---

## Key Learnings

1. **Convolution is the unifying operation** — most spatial filters reduce to sliding a kernel over the image.
2. **Kernel design determines behavior** — same framework yields blurring, sharpening, or edge detection depending on kernel values.
3. **From-scratch vs. library** — manual implementations expose floating-point considerations, boundary handling, and computational cost (especially for larger kernels).
4. **Filter selection depends on noise type** — Gaussian for Gaussian noise, Median for impulse noise, Bilateral when edges must be preserved.

---

## References

- Gonzalez, R. C. & Woods, R. E. — *Digital Image Processing*, 4th Edition
- OpenCV Documentation: [https://docs.opencv.org](https://docs.opencv.org)
- Pillow Documentation: [https://pillow.readthedocs.io](https://pillow.readthedocs.io)
- Harris, C. & Stephens, M. (1988). A combined corner and edge detector.

---

*Part of a series of ML/CV projects exploring foundations of machine learning and computer vision.*
