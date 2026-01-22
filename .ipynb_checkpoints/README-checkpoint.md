# Cassava Leaf Disease Classification

## Overview

Cassava is a critical staple crop for millions of people worldwide, yet its yield is highly vulnerable to leaf diseases that are difficult to diagnose accurately by visual inspection alone. This project focuses on automated cassava leaf disease classification using deep learning to support early and reliable disease detection.

The model is trained and validated using two cassava leaf disease datasets provided by the Makerere University AI Lab, which contain labeled images covering multiple disease categories and healthy leaves. To evaluate generalization performance, the system is tested on images from the iCassava 2019 Fine-Grained Visual Categorization Challenge, a widely used benchmark dataset for cassava disease recognition.

By combining datasets from different sources, this project aims to improve robustness to real-world variations such as lighting conditions, leaf orientation, and background noise. In addition to classification performance, the project incorporates model interpretability techniques (e.g., Grad-CAM) to visualize which regions of the leaf images influence the model’s predictions, helping to build trust and insight into the learned representations.

## Key Features

 - Multi-dataset training and evaluation.
 - Deep learning–based image classification
     - Uses convolutional neural networks to classify cassava leaf images into multiple disease categories, including healthy leaves, based on visual symptoms.
 - Improved robustness to real-world variation
    - Training on data from multiple sources helps the model handle variations in lighting, background, leaf orientation, and image quality commonly encountered in field conditions.
 - Model interpretability with visual explanations
     - Integrates Grad-CAM to generate class activation maps, highlighting the regions of the leaf that most strongly influence the model’s predictions.
 - Reproducible training and evaluation pipeline
     - Provides a structured workflow for data loading, training, validation, and testing to support reproducibility and further experimentation.
 - Extensible design
     - The project is structured to allow easy experimentation with different model architectures, training strategies, and interpretability methods.

## Usage

Requirements

 - Python 3.8+
 - Core packages: pandas, numpy, torch, matplotlib, scikit-learn

## Installation

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn torch
```

Download datasets from:
 - https://kaggle.com/competitions/cassava-leaf-disease-classification, 2020. Kaggle. (train images)
 - https://kaggle.com/competitions/cassava-disease (test images)

Extract only the train images, tfrecords, train.csv, label_num_to_disease_map from the first link. Extract only the train.zip from the test images folder, unzip it, and rename the folder to test-images. 
Place all files in: cassava-leaf-disease-classification-data/

## Results

 - The proposed network achieved approximately 52% classification accuracy on noisy, real-world cassava leaf images captured with accessible consumer cameras, demonstrating meaningful performance under realistic field conditions.
 - A patience-based early stopping strategy was employed during training to effectively reduce overfitting and improve generalization to unseen data.
 - Grad-CAM (Gradient-weighted Class Activation Mapping) was implemented to provide post-hoc interpretability, enabling visualization of spatial regions most influential to the model’s predictions.
 - Qualitative analysis of Grad-CAM heatmaps showed that the model consistently attended to disease-relevant leaf regions, supporting the validity of learned representations and offering insight into model successes and failure modes.

## Notes and next steps

 - Train and test accuracy using pretrained models such as ResNet50 or DenseNet
 - Use a pretrained model with Grad-CAM to not only show extensible design but return more precise results.
 - Increase the number of epochs and patience on my current model on better hardware to achieve most accurate results.
 - Use a brand new, more complicated image to effectively test the proposed model and better identify possible issues.

## Contact

ErnestMwebaze, Jesse Mostipak, Joyce, Julia Elliott, and Sohier Dane. Cassava Leaf Disease Classification. https://kaggle.com/competitions/cassava-leaf-disease-classification, 2020. Kaggle.

Mwebaze, Ernest, et al. “iCassava 2019 Fine-Grained Visual Categorization Challenge.” arXiv:1908.02900, arXiv, 24 Dec. 2019. arXiv.org, https://doi.org/10.48550/arXiv.1908.02900.

## License

- This codebase is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license