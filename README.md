# 🖼️ Image Clustering & Mobile Export Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-005CED.svg?style=flat&logo=ONNX&logoColor=white)](https://onnx.ai/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/facebook/dinov2-small)

An end-to-end machine learning pipeline that groups images by visual similarity using **DINOv2**, assigns semantic labels via **OpenAI CLIP**, and exports an optimized **INT8 Quantized ONNX** model for real-time mobile inference.

## 🚀 Key Features
- **Precision Clustering:** Uses DINOv2 (Vision Transformer) embeddings for high-fidelity visual grouping.
- **Semantic Naming:** Automatically labels clusters using CLIP and a curated 47-category gallery vocabulary.
- **Content Awareness:** Rule-based detection for screenshots, documents, and memes to separate digital clutter from photography.
- **Mobile Optimized:** Fully automated export path to 8-bit quantized ONNX, reducing model size by ~75% (from 86MB to 22MB).
- **Incremental Processing:** State management via `cluster_state.pkl` to allow adding new images without re-clustering the entire dataset.

## 🛠️ Technology Stack
| Component | Tool / Model |
| :--- | :--- |
| **Feature Extraction** | Facebook DINOv2 (Small) |
| **Clustering** | UMAP + K-Means (Silhouette Analysis) |
| **Labeling** | OpenAI CLIP (ViT-B/32) |
| **Optimization** | ONNX Runtime + INT8 Dynamic Quantization |
| **Visualization** | Matplotlib, Seaborn, Plotly |

## 📋 Pipeline Phases
1. **Environment Setup:** Dependency installation and GPU sanity checks.
2. **Feature Extraction:** Generating 384-dimensional visual fingerprints.
3. **Clustering:** Dimensionality reduction and mathematical determination of 'Optimal K'.
4. **Quality Analysis:** Generating heatmaps, scatter plots, and cluster contact sheets.
5. **Semantic Naming:** Multi-label classification and digital content filtering.
6. **Mobile Export:** Conversion to ONNX and performance benchmarking for Android/iOS.

## 📲 Mobile Integration Note
To use the exported `dinov2_small_embedder_int8.onnx` in your Android app:
1. Place the file in `app/src/main/assets/`.
2. Ensure your `OnnxEmbedder.kt` uses an input shape of `[1, 3, 224, 224]`.
3. The model outputs a normalized `[1, 384]` vector ready for cosine similarity comparison.

---
*Developed for high-performance gallery organization and smart device photo management.* 
