# SPPIFN

This is a ML project using the Kaggle dataset to train models on produce freshness prediction. Models take images as input, and output the predicted likelihood of the given produce being fresh. These models are trained with efficiency in mind, as they will be deployed on older generation apple iPads.

Project includes util files for:
    Converting trained model weights to CoreML (needed format for iOS)
    Data augmentation: increase the size of the dataset (different augmentations for images)
    Evaluate Dataset: eval pretrained and fine-tuned models on a dataset
    Inference: run inference on one image with one model
    Restructure by Freshness: created to fix the format of the dataset (fresh and rotten should be parent directories)

### Copyright Cade Stocker 2026