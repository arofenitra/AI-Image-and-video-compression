# Impact of AI Image and Video Compression on Deep Learning Tasks

## 📋 Project Overview

This comprehensive research project investigates how AI-based image and video compression methods affect the performance of various deep learning tasks including classification, object detection, OCR, and edge detection. We compare state-of-the-art neural compression models with traditional compression methods to understand their impact on downstream computer vision applications.

**Input image text detection**            |  **Compressed Image text detection**
:-------------------------:|:-------------------------:
![](https://github.com/ay-tishka/Impact-of-NIC-on-image-classification/blob/main/experiments/license%20analysis/uncomp_better_1.png)   |  ![](https://github.com/ay-tishka/Impact-of-NIC-on-image-classification/blob/main/experiments/license%20analysis/comp_worse_1.png)


## 🎯 Key Research Questions

- How do neural image compression (NIC) and neural video compression (NVC) models compare to traditional methods?
- What is the impact of compression on medical image diagnosis accuracy?
- Can compression actually improve OCR performance in certain scenarios?
- How does video compression affect object detection and edge detection tasks?

## 🔬 Key Contributions

- **Compression Model Evaluation**: Comprehensive comparison of Cheng2020-anchor and Cheng2020-attn NIC models against JPEG using PSNR, SSIM, VIF, and BPP metrics
- **Medical Image Analysis**: Analyzed performance degradation in skin lesion classification on compressed medical images (ISIC2018 dataset)
- **OCR Performance Study**: Discovered surprising improvements (~12.3%) in OCR accuracy for license plate recognition after compression
- **Object Detection Impact**: Extensive evaluation on COCO dataset showing varied compression effects across different object categories
- **Video Compression Analysis**: Investigation of video quality and task performance using ffmpeg and VIC models on MOT17 dataset
- **Edge Detection Study**: Comparison of HED and Canny edge detection on compressed video content

## 📊 Datasets

| Dataset | Task | Description |
|---------|------|-------------|
| **Kodak** | Compression Baseline | Standard image compression evaluation dataset |
| **ISIC2018** | Medical Classification | Skin lesion classification with high-resolution medical images |
| **US License Plates** | OCR Evaluation | License plate recognition via PaddleOCR |
| **COCO 2017** | Object Detection | Standard object detection benchmark |
| **MOT17** | Video Analysis | Multi-object tracking dataset for video compression evaluation |
| **BSDS500** | Edge Detection | Berkeley segmentation dataset for edge detection |
| **UCF101** | Video Classification | Action recognition in video sequences |

## 🏗️ Project Structure

```
AI-Image-and-video-compression/
├── dataset/
│   └── get_dataset.txt              # Dataset download instructions
├── docs/                            # Research papers and documentation
├── images/                          # Sample images and results
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── lib.py                       # Core library functions
└── experiments/
    ├── image/
    │   ├── image_classification/    # Medical image classification experiments
    │   ├── image_compression/       # Core compression experiments
    │   ├── image_object_detection/  # COCO object detection analysis
    │   └── image_OCR/              # License plate OCR experiments
    └── video/
        ├── video_classification/    # Action recognition experiments
        ├── video_compression/       # Video compression analysis
        ├── video_object_detection/  # MOT17 detection experiments
        └── video_edge_detection/    # Edge detection on compressed video
```

## 🚀 Quick Start

### Environment Setup

#### For Google Colab Users
```bash
pip install compressai
# Restart kernel after installation
```

#### For Local Installation
```bash
# Create virtual environment
python3 -m venv compressaienv
source compressaienv/bin/activate

# Install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Install CompressAI
git clone https://github.com/InterDigitalInc/CompressAI compressai
cd compressai
pip install wheel
python3 setup.py bdist_wheel --dist-dir dist/
pip install dist/compressai-*.whl
```

### Dataset Setup
```bash
# Navigate to dataset directory
cd dataset/

# Execute dataset download script
python get_datasets.py
```

## 🧪 Experiments

### 1. Image Compression Baseline
- **Location**: `experiments/image/image_compression/`
- **Models**: Cheng2020-anchor, Cheng2020-attn
- **Metrics**: PSNR, SSIM, VIF, BPP, MSE, MAE
- **Dataset**: Kodak
- **Key Finding**: Demonstrates rate-distortion trade-offs with quality parameter tuning

### 2. Medical Image Classification
- **Location**: `experiments/image/image_classification/`
- **Model**: DenseNet201 (pretrained + fine-tuned)
- **Task**: 9-class skin lesion classification
- **Dataset**: ISIC2018
- **Metrics**: Accuracy, F1-score, Cohen's Kappa
- **Key Finding**: Up to 20% accuracy degradation on compressed medical images

### 3. OCR Performance Analysis
- **Location**: `experiments/image/image_OCR/`
- **Tool**: PaddleOCR (PP-OCRv3)
- **Dataset**: US License Plates
- **Compression**: Cheng2020 models at Q1/Q3/Q6 quality levels
- **Key Finding**: Q3 compression improved OCR accuracy in ~12.3% of cases

### 4. Object Detection on Images
- **Location**: `experiments/image/image_object_detection/`
- **Model**: YOLOv5
- **Dataset**: COCO 2017
- **Key Finding**: Compression effects vary significantly across object categories

### 5. Video Compression Analysis
- **Location**: `experiments/video/video_compression/`
- **Model**: SSF2020 (Scale-Space Flow)
- **Dataset**: MOT17
- **Key Finding**: General accuracy reduction in object detection tasks

### 6. Video Edge Detection
- **Location**: `experiments/video/video_edge_detection/`
- **Models**: HED vs Canny Edge Detection
- **Datasets**: MOT17 (testing), BSDS500 (training)
- **Key Finding**: HED outperforms Canny on compressed video content

### 7. Video Classification
- **Location**: `experiments/video/video_classification/`
- **Dataset**: UCF101
- **Focus**: Action recognition performance on compressed video

## 📈 Key Results Summary

- **Medical Imaging**: Significant accuracy drops (up to 20%) highlight the need for specialized compression in medical applications
- **OCR Enhancement**: Counter-intuitively, moderate compression can improve OCR by reducing noise
- **Object Detection**: Compression impact varies by object type and complexity
- **Video Tasks**: Generally negative impact on detection accuracy, but HED edge detection shows resilience

## 🛠️ Technical Requirements

- Python 3.8+
- PyTorch 1.9+
- CompressAI
- OpenCV
- PaddleOCR
- YOLOv5
- Additional dependencies in `requirements.txt`

## 👥 Authors

- **QuocViet Pham**
- **Thanakrit Lerdmatayakul** 
- **Arofenitra Rarivonjy**

## 📚 Citations

```bibtex
@inproceedings{cheng2020image,
    title={Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules},
    author={Cheng, Zhengxue and Sun, Heming and Takeuchi, Masaru and Katto, Jiro},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2020}
}

@inproceedings{agustsson_scale-space_2020,
    title={Scale-Space Flow for End-to-End Optimized Video Compression},
    author={Agustsson, Eirikur and Minnen, David and Johnston, Nick and Balle, Johannes and Hwang, Sung Jin and Toderici, George},
    booktitle={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2020},
    pages={8500--8509}
}
```

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines and feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.