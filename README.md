# RGB–Infrared Image Registration Pipeline

This project implements a multi-stage alignment pipeline to register Infrared (IR) images with corresponding RGB images.

The objective is to achieve accurate multi-modal spatial correspondence for downstream computer vision tasks such as perception, detection, and sensor fusion.

---

## Pipeline Overview

The alignment is performed in three stages:

### Global Alignment

* Similarity transform using intensity-based registration (Mean Squares metric)
* Metadata synchronization (spacing, origin, direction)

### Local Geometric Refinement

* B-Spline based deformation for correcting regional misalignment

### Fine Non-Linear Warping

* Thin Plate Spline (TPS) interpolation using manually defined control points

---

## Features

* Multi-modal RGB–IR registration
* Global + local transformation modeling
* Batch processing pipeline
* Overlay visualization for alignment validation

---

## Technologies Used

* Python
* SimpleITK
* OpenCV
* NumPy
* SciPy

---

## Sample Alignment Result

![RGB IR Alignment](alignment_overlay.png)


