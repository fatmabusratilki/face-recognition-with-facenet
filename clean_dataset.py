import os
import shutil
import cv2
import pandas as pd
import random
from collections import defaultdict
from tqdm import tqdm

# Clean directories based on image count
def clean_directories_by_image_count(base_dir, max_images=20):
    for person in os.listdir(base_dir):
        person_path = os.path.join(base_dir, person)
        if os.path.isdir(person_path):
            images = os.listdir(person_path)
            if len(images) < max_images:
                shutil.rmtree(person_path)
                print(f"Deleted (too few images): {person_path}")
            elif len(images) > max_images:
                images = sorted(images)
                for img in images[max_images:]:
                    img_path = os.path.join(person_path, img)
                    os.remove(img_path)
                    print(f"Deleted (extra image): {img_path}")

# Remove folders without detectable faces
def contains_face(image_path, face_cascade):
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def remove_folders_without_faces(base_dir, face_cascade):
    for person in os.listdir(base_dir):
        person_path = os.path.join(base_dir, person)
        if os.path.isdir(person_path):
            images = os.listdir(person_path)
            has_face = any(contains_face(os.path.join(person_path, img), face_cascade) for img in images)
            if not has_face:
                shutil.rmtree(person_path)
                print(f"Deleted (no face detected): {person_path}")

# Balance the dataset by gender and split
def balance_dataset(csv_path, input_root, output_root, split_config):
    df = pd.read_csv(csv_path)
    df.set_index("image_id", inplace=True)

    for split, targets in split_config.items():
        print(f"\n🔍 Processing split: {split.upper()}")
        gender_groups = {"female": [], "male": []}
        base_folder = os.path.join(input_root, split)

        for person in os.listdir(base_folder):
            person_path = os.path.join(base_folder, person)
            if not os.path.isdir(person_path):
                continue

            images = [f for f in os.listdir(person_path) if f.endswith(".jpg")]
            gender_votes = []

            for img in images:
                if img in df.index:
                    gender_votes.append(df.loc[img]["Male"])

            if gender_votes:
                majority_gender = pd.Series(gender_votes).mode()[0]
                gender = "female" if majority_gender == -1 else "male"
                gender_groups[gender].append(person)

        for gender in ["female", "male"]:
            random.shuffle(gender_groups[gender])
            selected_people = gender_groups[gender][:targets[gender]]

            output_split_dir = os.path.join(output_root, split)
            os.makedirs(output_split_dir, exist_ok=True)

            for person in tqdm(selected_people, desc=f"{split}-{gender}"):
                src_path = os.path.join(base_folder, person)
                dst_path = os.path.join(output_split_dir, person)
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)

    print("\nSuccessfully copied people while preserving folder structure.")

# 🔧 MAIN EXECUTION
if __name__ == "__main__":
    input_root = "CelebA/processed"
    output_root = "celeba_balanced"
    csv_path = "CelebA/list_attr_celeba.csv"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    split_config = {
        "train": {"female": 400, "male": 400},
        "val": {"female": 50, "male": 50},
        "test": {"female": 50, "male": 50},
    }

    for split in ["train", "val", "test"]:
        print(f"\nCleaning and checking face data for {split.upper()}")
        base_dir = os.path.join(input_root, split)
        clean_directories_by_image_count(base_dir, max_images=20)
        remove_folders_without_faces(base_dir, face_cascade)

    balance_dataset(csv_path, input_root, output_root, split_config)
