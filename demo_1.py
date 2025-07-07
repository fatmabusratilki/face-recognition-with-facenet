import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from datetime import datetime
from PIL import Image
import threading
import tensorflow as tf
from tensorflow.keras import layers, models
from mtcnn import MTCNN

# ========== MODEL YÜKLEME ==========
def FaceNet(embedding_size=128):
    inputs = tf.keras.Input(shape=(160, 160, 3))

    x = layers.Conv2D(64, (3, 3), padding='valid', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(embedding_size, activation=None)(x)

    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)
    return models.Model(inputs, outputs)

# ========== Model Yükleme ==========
model_path = "fine_tune_best_model_same_data.keras"
model = FaceNet(embedding_size=256)
model.load_weights(model_path)
# model = tf.keras.models.load_model(model_path, custom_objects={'Lambda': tf.keras.layers.Lambda, 'Facenet': FaceNet}, compile=False)
detector = MTCNN()

def load_model_from_path(path):
    global model
    model = tf.keras.models.load_model(path)
    model.summary()

# ========== ÖN İŞLEME ==========
def preprocess(img):
    img = tf.image.resize(img, (160, 160))
    img = (img / 255.0) #- 1.0  # [-1, 1] aralığına normalize #Burdaki -1.0 değerini değiştirerek normalizasyonu ayarlayabilirsin
    return tf.expand_dims(img, axis=0)

# ========== EMBEDDING HESAPLAMA ==========
def get_embedding(face_img):
    face = preprocess(face_img)
    emb = model.predict(face, verbose=0)[0]
    return emb

# ========== KİŞİ EKLE ==========
def add_new_person():
    name = simpledialog.askstring("Yeni Kişi", "Kişinin adını ve soyadını girin:")
    if not name: return

    os.makedirs("known_person", exist_ok=True)
    person_dir = os.path.join("known_person", name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0
    messagebox.showinfo("Yüz Kaydı", "Kameraya bakın. 5 yüz görüntüsü kaydedilecek.")

    while count < 5:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(rgb)
        if result:
            x, y, w, h = result[0]['box']
            face = rgb[y:y+h, x:x+w]
            if face.size == 0: continue

            face_img = Image.fromarray(face)
            filename = os.path.join(person_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            face_img.save(filename)
            count += 1
            print(f"📸 {count}/5 yüz görüntüsü kaydedildi.")
        cv2.imshow("Yüz Kaydı", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()
    if count > 0:
        messagebox.showinfo("Tamamlandı", f"{name} için {count} yüz kaydedildi.")
    else:
        messagebox.showwarning("Başarısız", "Yüz algılanamadı.")

def delete_person_gui():
    name = simpledialog.askstring("Kişi Sil", "Silmek istediğiniz kişinin adını girin:")
    if name:
        folder_path = os.path.join("known_person", name)
        if os.path.exists(folder_path):
            if messagebox.askyesno("Emin misiniz?", f"{name} silinsin mi?"):
                import shutil
                shutil.rmtree(folder_path)
                messagebox.showinfo("Silindi", f"{name} başarıyla silindi.")
        else:
            messagebox.showerror("Hata", f"{name} bulunamadı.")


# ========== VERİ YÜKLE ==========
def load_known_faces():
    known_embeddings = {}
    for person in os.listdir("known_person"):
        person_path = os.path.join("known_person", person)
        if not os.path.isdir(person_path): continue

        embeddings = []
        for img_file in os.listdir(person_path):
            try:
                img_path = os.path.join(person_path, img_file)
                img = Image.open(img_path).convert("RGB")
                img = np.array(img)
                emb = get_embedding(img)
                embeddings.append(emb)
            except Exception as e:
                print(f"Hata: {e}")
                continue

        if embeddings:
            known_embeddings[person] = np.mean(embeddings, axis=0)
    return known_embeddings

# ========== YÜZ TANIMA ==========
def recognize_faces():
    known_faces = load_known_faces()
    if not known_faces:
        messagebox.showwarning("Veri Yok", "Kayıtlı yüz bulunamadı!")
        return

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)
        for result in results:
            x, y, w, h = result['box']
            x, y = max(x, 0), max(y, 0)
            x2 = min(x + w, frame.shape[1])
            y2 = min(y + h, frame.shape[0])
            face = rgb[y:y2, x:x2]
            # face = rgb[y:y+h, x:x+w]
            if face.size == 0: continue

            try:
                emb = get_embedding(face)
            except:
                continue

            known_faces = load_known_faces()
            
            best_match = "Bilinmiyor"
            min_dist = float('inf')
            for name, ref_emb in known_faces.items():
                dist = np.linalg.norm(emb - ref_emb)
                if dist < min_dist:
                    min_dist = dist
                    best_match = name

            if min_dist < 0.7:
                label = f"{best_match}"
                color = (0, 255, 0)
            else:
                label = "Bilinmiyor"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Yüz Tanima", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

# ========== TKINTER ARAYÜZ ==========

def start_recognition_thread():
    threading.Thread(target=recognize_faces).start()

root = tk.Tk()
root.title("Face Recognition System - TensorFlow")
root.geometry("450x400")
root.configure(bg="#2b2b2b")

# Başlık
tk.Label(
    root, 
    text="Face Recognition System", 
    font=("Helvetica", 22, "bold"), 
    fg="#00ffcc", 
    bg="#2b2b2b"
).pack(pady=30)

# Butonları içeren çerçeve
button_frame = tk.Frame(root, bg="#2b2b2b")
button_frame.pack(pady=10)

def create_button(text, command, pady=10):
    return tk.Button(
        button_frame,
        text=text,
        font=("Arial", 14),
        width=25,
        bg="#444", fg="white",
        activebackground="#00aa88",
        activeforeground="white",
        relief="raised",
        bd=2,
        command=command
    ).pack(pady=pady)

# Butonlar
create_button("Start Face Recognition", start_recognition_thread)
create_button("Add New Person", add_new_person)
create_button("Delete Person", delete_person_gui)
create_button("EXIT", root.destroy)

root.mainloop()

# # ========== TKINTER ARAYÜZ ==========
# def start_recognition_thread():
#     threading.Thread(target=recognize_faces).start()

# root = tk.Tk()
# root.title("Yüz Tanıma Sistemi - TensorFlow")
# root.geometry("400x300")
# root.configure(bg="#1e1e1e")

# tk.Label(root, text="Yüz Tanıma Sistemi", font=("Arial", 18), fg="white", bg="#1e1e1e").pack(pady=30)
# tk.Button(root, text="Yüz Tanımaya Başla", font=("Arial", 14), command=start_recognition_thread).pack(pady=20)
# tk.Button(root, text="Yeni Kişi Ekle", font=("Arial", 14), command=add_new_person).pack(pady=10)
# tk.Button(root, text="Kişi Sil", font=("Arial", 12), command=delete_person_gui).pack(pady=5)
# tk.Button(root, text="Çıkış", font=("Arial", 12), command=root.destroy).pack(pady=10)


root.mainloop()
