import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.keras.preprocessing import image
import os
import pandas as pd

def preprocess_image(image_path):
    """Görüntüyü uygun boyuta getir ve normalizasyon uygula."""
    img = image.load_img(image_path, target_size=(160, 160))  
    img = image.img_to_array(img)  
    img = np.expand_dims(img, axis=0)  
    img = img.astype("float32")  
    return img

def evaluate_model(model, base_dir, csv_path, threshold=0.4):
    """Modelin performansını test verisi ile değerlendir."""
    
    # Test veri setinden çiftleri yükle
    df = pd.read_csv(csv_path)

    img1_paths = df['img1'].tolist()
    img2_paths = df['img2'].tolist()
    labels = df['label'].astype(int).tolist()  # Labels'ı integer'a çevir


    predictions = []
    distances = []

    for img1_rel, img2_rel in zip(img1_paths, img2_paths):
        img1 = preprocess_image(os.path.join(base_dir, img1_rel))
        img2 = preprocess_image(os.path.join(base_dir, img2_rel))

        emb1 = model.predict(img1, verbose=0).flatten()
        emb2 = model.predict(img2, verbose=0).flatten()

        dist = euclidean_distances([emb1], [emb2])[0][0]

        distances.append(dist)
        predictions.append(1 if dist < threshold else 0)

    # Performans metriklerini hesapla
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=1)
    rec = recall_score(labels, predictions, zero_division=1)
    f1 = f1_score(labels, predictions, zero_division=1)
    fpr, tpr, _ = roc_curve(labels, -np.array(distances), pos_label=1)  # Negatif mesafe skoru kullan
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


model = tf.keras.models.load_model("model.h5", custom_objects={'Lambda': tf.keras.layers.Lambda}, compile=True)
evaluate_model(model, base_dir='datasets/test', csv_path='pairs.csv', threshold=0.4)