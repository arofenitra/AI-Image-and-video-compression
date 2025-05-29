# AI-Image-and-video-compression
## Project Description
This project investigates on how AI-based image and video compression affect several deep learning tasks such as classification, object detection, and edge detection. The project will use a dataset of images and videos, and will apply traditional-based compression methods such as JPEG 2000, JPEG, and deep learning-based compression methods such as autoencoders. 

The project will compare the performance of AI-based compression methods with traditional compression methods such as image and video compression. The project will also explore the impact of compression on the quality of the compressed images and videos.

## Project Overview

Traditional compression methods (e.g., JPEG, ffmpeg) often sacrifice subtle features important for classification. This project compares conventional methods with state-of-the-art NIC and NVC models, focusing on their impact on downstream tasks like medical image diagnosis and license plate recognition.

Traditional compression methods (e.g., JPEG) often sacrifice subtle features important for classification. This project compares conventional methods with state-of-the-art NIC models, focusing on their impact on downstream tasks like medical image diagnosis and license plate recognition.

**Input image text detection**            |  **Compressed Image text detection**
:-------------------------:|:-------------------------:
![](https://github.com/ay-tishka/Impact-of-NIC-on-image-classification/blob/main/experiments/license%20analysis/uncomp_better_1.png)   |  ![](https://github.com/ay-tishka/Impact-of-NIC-on-image-classification/blob/main/experiments/license%20analysis/comp_worse_1.png)


## Key Contributions
- Compared Cheng2020-anchor and Cheng2020-attn NIC models against JPEG using PSNR, SSIM, VIF, and BPP.
- Analyzed performance degradation in classification on compressed medical images (ISIC2018 dataset).
- Explored surprising improvements in OCR accuracy for license plate recognition after compression.
- Explored study on object detection from Coco Datasets.
- Investigated the impact of compression on video quality using ffmpeg and VIC models.
- Imapct of video compression on object detection, video classification and edge detection.

## Datasets Used
- Kodak – Used for evaluating compression performance as a baseline.
- ISIC2018 – Skin lesion classification with high-resolution medical images.
- US License Plates – For OCR evaluation via PaddleOCR.
- Coco Dataset – For object detection evaluation.
- Video Dataset : MOT17 – For video compression and quality evaluation.
- BSDS500 – For edge detection evaluation.
- UCF101 – For video classification evaluation.
# User guide

## CompressAI For Colab users:
```bash
pip install compressai
```
It will require the notebook to be restarted for once.

## CompressAI Local Installation Guide

```bash
python3 -m venv compressaienv
source compressaienv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio
python3 -m pip install matplotlib pandas
python3 -m pip install numpy scipy
python3 -m pip install ipykernel
python3 -m pip install jupyter
python3 -m ipykernel install --user --name=compressaienv
python3 -m pip install pybind11
module load compilers/gcc-8.3.0
```

Clone [CompressAI](https://github.com/InterDigitalInc/CompressAI) 
```
git clone https://github.com/InterDigitalInc/CompressAI compressai
cd compressai
pip install wheel
python3 setup.py bdist_wheel --dist-dir dist/
pip install dist/compressai-*.whl
```


## Experiments Summary
**0. Testing Cheng2020-Anchor and Cheng2020-Attn AI-based compression model for Kodak dataset**
- Task: Investigate models' performance
- Metric: PSNR, SSIM, VIF, BPP, MSE, MAE
- Dataset: https://www.kaggle.com/datasets/sherylmehta/kodak-dataset
- Notable finding: Rate-distortion trade-off by increasing the quality of compression model
- Running: `experiments/AICompression.ipynb`
  
**1.Skin Lesion Classification Summary**

- **Model**: DenseNet201 (pretrained + custom)
- **Task**: 9-class skin lesion classification
- **Data**: [ISIC Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic/data)  
  → Place `Train` and `Test` in `experiments/image_compression/ISIC-skin-cancer`
- **Weights**: [`skin_disease_model.h5`](https://www.kaggle.com/code/muhammadsamarshehzad/skin-cancer-classification-densenet201-99-acc/output) → Save in `experiments/`
- **Metrics**: Accuracy, F1, Cohen’s Kappa
- **Finding**: Up to 20% accuracy drop on compressed images
- **Note**: Run `AICompression.ipynb` first to generate degraded test images before using `DenseNet121_Aug_Clf (2).ipynb`


<!---
- Classifier: DenseNet201 (pretrained and custom-trained)
- Task: Skin lesion classification (9-class)
- Datasets: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic/data

  Make sure that Test and Train directories lie in `experiments/image_compression/ISIC-skin-cancer`
- Pretrained model for test in `experiments/AICompression.ipynb`: https://www.kaggle.com/code/muhammadsamarshehzad/skin-cancer-classification-densenet201-99-acc/output

  Make sure downloaded `skin_disease_model.h5` file lies in `experiments` directory.
- Metric: Accuracy, F1 score, Cohen’s Kappa
- Notable finding: Up to 20% drop in accuracy on compressed images vs original
- Remark: In order to run notebook `experiments/image_classification/DenseNet121_Aug_Clf (2).ipynb`, first run through Experiment 2 in `experiments/AICompression.ipynb` in order to generate degraded test images. Those images will lie in specific folder `decompressed` inside each label folder of test dataset.
-->



**2. OCR on Compressed Images**

- **Tool**: PaddleOCR (PP-OCRv3)
- **Setup**: US license plate dataset, Cheng2020-anchor & attn, Q1/Q3/Q6
- **Finding**: Q3 compression improved OCR in **~12.3%** of cases
- **Insight**: Anchor models outperformed; compression smoothed noise, boosted clarity

**3. Object Detection on Compressed Image**

- **Tool**: YOLOv5
- **Setup**: COCO dataset, cheng2020-anchor & attn, Q1/Q3/Q6
- **Finding**: Some labels is highly affected by compression, while some are not.
- **Insight**: Compression reduced noise, enhanced object clarity

**4. Object Detection on Compressed Video**
- **Tool**: ssf2020 for video compression
- **Setup**: MOT17 for the dataset and ssf2020 for the video compression model.
- **Finding**: Compression reduced detection accuracy in most of the cases.


**5. Edge Detection on Image and Compressed Video**
- **Tool**: ssf2020 for video compression
- **Setup**: MOT17 for the dataset testing and BSDS for the edge detection training dataset. ssf2020 for the video compression model.
- **Finding**: HED models perform better than Canny Edge Detection for original-compressed video.


## Authors
QuocViet Pham, Thanakrit Lerdmatayakul, Arofenitra Rarivonjy


## Related works
```bibtex
@inproceedings{ballemshj18,
  author    = {Johannes Ball{\'{e}} and
               David Minnen and
               Saurabh Singh and
               Sung Jin Hwang and
               Nick Johnston},
  title     = {Variational image compression with a scale hyperprior},
  booktitle = {6th International Conference on Learning Representations, {ICLR} 2018,
               Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings},
  publisher = {OpenReview.net},
  year      = {2018},
}
@inproceedings{minnenbt18,
  author    = {David Minnen and
               Johannes Ball{\'{e}} and
               George Toderici},
  editor    = {Samy Bengio and
               Hanna M. Wallach and
               Hugo Larochelle and
               Kristen Grauman and
               Nicol{\`{o}} Cesa{-}Bianchi and
               Roman Garnett},
  title     = {Joint Autoregressive and Hierarchical Priors for Learned Image Compression},
  booktitle = {Advances in Neural Information Processing Systems 31: Annual Conference
               on Neural Information Processing Systems 2018, NeurIPS 2018, 3-8 December
               2018, Montr{\'{e}}al, Canada},
  pages     = {10794--10803},
  year      = {2018},
}
@inproceedings{cheng2020image,
    title={Learned Image Compression with Discretized Gaussian Mixture
    Likelihoods and Attention Modules},
    author={Cheng, Zhengxue and Sun, Heming and Takeuchi, Masaru and Katto,
    Jiro},
    booktitle= "Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR)",
    year={2020}
}

@inproceedings{agustsson_scale-space_2020,
    title={Scale-{Space} {Flow} for {End}-to-{End} {Optimized} {Video}
    {Compression}},
    author={Agustsson, Eirikur and Minnen, David and Johnston, Nick and
    Balle, Johannes and Hwang, Sung Jin and Toderici, George},
    booktitle={2020 {IEEE}/{CVF} {Conference} on {Computer} {Vision} and
    {Pattern} {Recognition} ({CVPR})},
        publisher= {IEEE},
    year={2020},
        month= jun,
         year= {2020},
         pages= {8500--8509},
}```
