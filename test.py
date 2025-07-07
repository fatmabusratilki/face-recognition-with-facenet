import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Lambda
from tqdm import tqdm
from tensorflow.keras import layers, models, regularizers
from facenet import FaceNet

# Load model
model = tf.keras.models.load_model("last_model_finetuned_best_model.keras", custom_objects={'Lambda': tf.keras.layers.Lambda, 'Facenet': FaceNet}, compile=False)

test_data_dir = 'dataset/processed/test'

test_datagen = ImageDataGenerator(rescale=1./255)
test_dataset = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(160, 160),
    batch_size=64,
    class_mode='binary',
    shuffle=False
)

def preprocess_image(image_path):
    img = Image.open(image_path).resize((160, 160))
    img = img.convert('RGB')  # If you think it can be grayscale
    img = np.asarray(img).astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def create_image_pairs(celeba_dir, n_pairs=100):
    people = os.listdir(celeba_dir)
    people = [p for p in people if os.path.isdir(os.path.join(celeba_dir, p))]
    pairs = []
    labels = []

    for _ in range(n_pairs):
        # Positive example: two different photos of the same person
        person = np.random.choice(people)
        images = os.listdir(os.path.join(celeba_dir, person))
        if len(images) >= 2:
            img1, img2 = np.random.choice(images, 2, replace=False)
            pairs.append([
                os.path.join(celeba_dir, person, img1),
                os.path.join(celeba_dir, person, img2)
            ])
            labels.append(1)

        # Negative sample: one photo from different people
        p1, p2 = np.random.choice(people, 2, replace=False)
        img1 = np.random.choice(os.listdir(os.path.join(celeba_dir, p1)))
        img2 = np.random.choice(os.listdir(os.path.join(celeba_dir, p2)))
        pairs.append([
            os.path.join(celeba_dir, p1, img1),
            os.path.join(celeba_dir, p2, img2)
        ])
        labels.append(0)

    return pairs, labels


def find_best_threshold(model, image_pairs, labels, method='f1'):
    scores = []
    for img1_path, img2_path in tqdm(image_pairs, desc="Calculating distances"):
        img1 = preprocess_image(img1_path)
        img2 = preprocess_image(img2_path)
        emb1 = model.predict(img1, verbose=0)
        emb2 = model.predict(img2, verbose=0)
        dist = np.linalg.norm(emb1 - emb2, axis=1)[0]
        scores.append(dist)

    thresholds = np.linspace(min(scores), max(scores), 100)
    best_metric = -1
    best_threshold = thresholds[0]

    for threshold in thresholds:
        preds = [1 if s < threshold else 0 for s in scores]
        
        if method == 'f1':
            metric = f1_score(labels, preds)
        elif method == 'accuracy':
            metric = accuracy_score(labels, preds)
        elif method == 'precision':
            metric = precision_score(labels, preds)
        elif method == 'recall':
            metric = recall_score(labels, preds)
        else:
            raise ValueError("Unsupported method. Use 'f1', 'accuracy', 'precision', or 'recall'.")

        if metric > best_metric:
            best_metric = metric
            best_threshold = threshold

    print(f"\nBest threshold by {method}: {best_threshold:.4f} (score: {best_metric:.4f})")
    return best_threshold, scores

# Model evaluation
def evaluate_verification(model, image_pairs, labels, threshold, precomputed_scores=None):
    if precomputed_scores is None:
        scores = []
        for img1_path, img2_path in tqdm(image_pairs, desc="Calculating distances"):
            img1 = preprocess_image(img1_path)
            img2 = preprocess_image(img2_path)
            emb1 = model.predict(img1, verbose=0)
            emb2 = model.predict(img2, verbose=0)
            dist = np.linalg.norm(emb1 - emb2, axis=1)[0]
            scores.append(dist)
    else:
        scores = precomputed_scores

    predictions = [1 if s < threshold else 0 for s in scores]

    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    fpr, tpr, _ = roc_curve(labels, -np.array(scores), pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"ROC AUC      : {roc_auc:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

# Main stream
pairs, labels = create_image_pairs(test_data_dir, n_pairs=190)
best_threshold, scores = find_best_threshold(model, pairs, labels, method='accuracy')
metrics = evaluate_verification(model, pairs, labels, threshold=best_threshold, precomputed_scores=scores)
