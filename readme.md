# Lung Disease Prediction from Chest X-Ray Images Using Lightweight CNNs

This repository contains the implementation of models for predicting lung diseases, including COVID-19, pneumonia, and normal cases from chest X-ray images. The project explores the effectiveness of lightweight Convolutional Neural Networks (CNNs), specifically MobileNetV1 and MobileNetV2, enhanced with Squeeze-and-Excitation (SE) blocks for efficient and accurate diagnosis.

## Project Overview

The COVID-19 pandemic highlighted the need for fast, accurate, and accessible diagnostic methods. This project leverages lightweight deep learning models enhanced with attention mechanisms (Squeeze-and-Excitation blocks) to analyze chest X-ray (CXR) images and predict the presence of lung diseases. Techniques such as Class Activation Mapping (CAM) and Grad-CAM are used to ensure the interpretability of the model's decision-making process.

![cam_animation](dataset/figures/cam_animation.gif)

## Repository Structure

``` plaintext
project_root/
├── models/
│   ├── weights/              # Model weights
│   ├── mobilenet_v1.py       # MobileNetV1 model implementation
│   ├── mobilenet_v2.py       # MobileNetV2 model implementation
│   ├── se_net.py             # SE-enhanced MobileNet architectures
│   └── utils.py              # Utility functions for model processing
├── dataset/
│   ├── covid19_dataset/      # Dataset organized into categories (COVID, normal, pneumonia)
│   │   ├── covid/
│   │   ├── normal/
│   │   └── pneumonia/
│   ├── utils.py              # Dataset utility functions for loading and preprocessing
│   └── corrupted_images.txt  # Log of corrupted images removed from the dataset
├── notebooks/
│   ├── Models.ipynb          # Exploration and implementation of models
│   ├── Dataset.ipynb         # Dataset exploration and preparation pipeline
│   └── Experiments.ipynb     # Model training, evaluation, and experiments
├── readme.md                 # Readme file
├── results.csv               # Results table
└── project_report.pdf        # Project report describing the study in detail
```

## Notebooks Overview

-   `Dataset.ipynb`: In this notebook, we outline the pipeline we followed for dataset exploration and the creation of all necessary functions to prepare it for model training. All the functions defined here have been collected in the file `dataset/utils.py` to make them available and usable in the experiments conducted in the file `Experiments.ipynb`.

-   `Models.ipynb`: This notebook serves as a guide to the models we developed for this project. We have organized the implementation of these models into several Python scripts, each located in the `models` directory.

-   `Experiments.ipynb`: This notebook documents the experiments conducted for training and evaluating our models on the COVID-19 chest X-ray dataset. It consists of various sections, including dataset loading, training protocol, and model training using different MobileNet variants. We also perform hyperparameter tuning and analyze the trade-offs between accuracy and computational complexity. Finally, we visualize the model's decision-making process using CAM (Class Activation Mapping) and GradCAM (Gradient-weighted Class Activation Mapping) to highlight the regions of the chest X-ray images that the models focus on during predictions.

## Models

-   **MobileNetV1**: A lightweight CNN designed for mobile applications, utilizing depthwise separable convolutions to reduce computational costs.
-   **MobileNetV2**: Builds on MobileNetV1 with inverted residuals and linear bottlenecks, further improving efficiency and accuracy.
-   **SE-Enhanced MobileNets**: Both MobileNetV1 and MobileNetV2 architectures are enhanced with Squeeze-and-Excitation (SE) blocks. SE blocks recalibrate channel-wise feature responses, improving the model's ability to focus on relevant features in chest X-ray images.

## Dataset

The dataset used for this project consists of chest X-ray images classified into three categories: COVID-19, pneumonia, and normal cases. The data is sourced from various repositories, and images are preprocessed to ensure uniformity (224x224 pixels). Data augmentation techniques such as zooming, rotation, translation, and intensity shifting are applied to enhance model robustness.

## Experiments

The experiments were conducted using various configurations of MobileNetV1, MobileNetV2, and SE-enhanced versions of both. We explored different width multipliers to assess the trade-offs between model accuracy and computational efficiency. Additionally, we applied Class Activation Mapping (CAM) and GradCAM to visualize and interpret model decisions, ensuring the model focuses on clinically relevant regions of the X-rays during predictions.

## Results

-   The SE-enhanced MobileNetV1 and MobileNetV2 models demonstrated high accuracy in predicting lung diseases from chest X-rays.
-   SE-enhanced models showed improved diagnostic performance, especially in detecting subtle patterns associated with COVID-19 and pneumonia.
-   Visualizations using CAM and GradCAM provided insights into the model's focus areas, improving trust in the diagnostic process.

## How to Use

1.  Create the dataset:

    -   Download [dataset](https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/jctsfj2sfn-1.zip) and extract the content of the zip file in the `/dataset/covid19-dataset/` directory.

2.  Get pretrained models:

    -   Download [pretrained models](https://drive.google.com/drive/folders/1qhMSy1m0f2SXews_VLXH3fNm_tO_v1K5?usp=sharing) and move the `.keras` file into the `/models/weights/` directory.

3.  Explore the dataset and models:

    -   Run the notebooks in the `notebooks/` folder to see the dataset preparation, model implementation, and experiments.
