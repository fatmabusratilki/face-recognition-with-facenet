# Face Recognition System with FaceNet 

This project implements a deep learning-based facial recognition system using MTCNN for face detection and FaceNet for face embedding. The system is capable of identifying individuals based on facial similarity using Euclidean distance.

## Project Overview

The system takes a facial image as input, detects the face using MTCNN, and generates a 128-dimensional embedding vector with FaceNet. Then, this embedding is compared to a database of known embeddings to identify the person using a predefined distance threshold.

## Features

- Face detection with **MTCNN**
- Face embedding with **FaceNet**
- Uses **triplet loss** for training
- Supports **real-time recognition**
- Handles **occlusion and profile variations**
- Custom balanced dataset created from CelebA
- Achieves **85% accuracy** on a private test set

## Methods Used

### 1. **Face Detection (MTCNN)**
- Detects face regions in input images.
- Scales detected faces to 160x160 for FaceNet.
- Extracts landmarks and bounding boxes.

### 2. **Face Embedding (FaceNet)**
- Projects faces into a 128D Euclidean space.
- Uses L2 normalization and triplet loss.
- Distinguishes identities by comparing vector distances.

### 3. **Triplet Loss**
- Ensures:  
  `distance(anchor, positive) + margin < distance(anchor, negative)`
- Uses semi-hard negative mining after initial epochs for better generalization.

## Dataset

### Training Set
- Based on CelebA dataset with 10,177 identities and 202,599 images.
- Balanced to 800 identities, 20 images per identity.
- Gender and image attribute (e.g., glasses) distribution considered.

### Validation & Test Sets
- Validation: 100 identities, 20 images per identity.
- Test: 40 identities, 5 images each from personal & internet sources.
- Designed to reflect real-world challenges (occlusion, profile views, etc.).

## Model Architecture

- Input size: `160x160x3`
- 5 convolutional layers + BatchNorm + MaxPooling
- Flatten → Dense layers → Dropout
- Final embedding output normalized with L2

## Hardware and Software

| Component      | Description                   |
|----------------|-------------------------------|
| GPU            | NVIDIA GeForce RTX 4070 (Laptop) |
| Framework      | TensorFlow (with CUDA)         |
| Language       | Python 3.9                     |
| Libraries      | scikit-learn, matplotlib       |

## Training Process

- Optimizer: **RMSprop** (after testing with Adam, Adagrad, SGD)
- Best hyperparameters:  
  - `batch_size = 32`  
  - `learning_rate = 1e-4` with **Cosine Decay Scheduler**
- Uses **early stopping** with `patience = 5`

## Results

- Test set size: 380 images (190 positive + 190 negative pairs)
- Accuracy: **85%**
- Confusion matrix and ROC curve used for evaluation
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)
## Example Use Cases

- Verify identity from facial photo
- Compare unknown image to database
- Real-time webcam recognition support

## Future Work

- Train on larger datasets (e.g., VGGFace2)
- Improve generalization to unseen environments
- Optimize for edge deployment (e.g., Jetson Nano)

## Getting Started

### Dataset Preparation (CelebA)

1. **Download CelebA dataset**  
   Get the dataset from Kaggle:  
    https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

2. **Split the dataset**  
   Run `dataset_sep.py` to separate the original dataset into `train`, `val`, and `test` folders:
   ```bash
   python dataset_sep.py

3. **Preprocess Images**
   Detect faces using MTCNN and resize them to 160x160 (FaceNet input size):
   ```bash
   python preprocess_data.py
   
4. **Clean the Dataset**
   ```bash
   python clean_dataset.py

### Model Training
1. **Train with Random Triplets**
   Start the initial training phase with randomly selected triplets:
   ```bash
   python train.py

2. **Fine-Tune with Semi-Hard Negative Triplets**
   For better discrimination between difficult cases, fine-tune the model using semi-hard negatives:
   ```bash
   python facenet_train_semihard.py

#### Training History
Below is the accuracy and loss progression during training:
![alt text](image.png)


## Real-Time Demo
After training the model or downloading the pretrained model (`model.zip`), you can run real-time face recognition using your webcam:
    ```bash
    python demo.py












   

