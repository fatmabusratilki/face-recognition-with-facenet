import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class GetTriplets(tf.keras.utils.Sequence):
    def __init__(self, image_folder_dataset, transform=None, batch_size=32):
        self.transform = transform
        self.image_folder_dataset = image_folder_dataset
        self.batch_size = batch_size
        self.data = self.create_triplets()

    def create_triplets(self):
        person_dict = {}
        
        filepaths = self.image_folder_dataset.filepaths  # Görüntü yolları
        labels = self.image_folder_dataset.classes # Class labels for each image

        # Group images by their labels
        for img_path, label in zip(filepaths, labels):
            if label not in person_dict:
                person_dict[label] = []
            person_dict[label].append(img_path)

        
        triplets = []
        class_labels = list(person_dict.keys())

        # Create triplets

        for label, images in person_dict.items():
            if len(images) < 2:
                continue

            for anchor in images:
                positive = random.choice([img for img in images if img != anchor])
                
                # Random negative selection
                negative_label = random.choice([l for l in labels if l != label])
                negative = random.choice(person_dict[negative_label])
                
                triplets.append((anchor, positive, negative))
        
        return triplets
    
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        batch_triplets = self.data[index * self.batch_size : (index + 1) * self.batch_size]
        
        anchor_imgs, positive_imgs, negative_imgs = [], [], []
        
        for anchor_path, positive_path, negative_path in batch_triplets:
            anchor_img = self.load_image(anchor_path)
            positive_img = self.load_image(positive_path)
            negative_img = self.load_image(negative_path)
            
            anchor_imgs.append(anchor_img)
            positive_imgs.append(positive_img)
            negative_imgs.append(negative_img)
        
        return [np.array(anchor_imgs), np.array(positive_imgs), np.array(negative_imgs)]
    
    def load_image(self, img_path):
        img = image.load_img(img_path, target_size=(160, 160))  # Resize if necessary
        img = image.img_to_array(img) / 255.0  # Normalize
        
        if self.transform: 
            img = self.transform(img)
        
        return img


# # Veri yolu
# data_dir = "dataset/processed/train"  # Görsellerinizin bulunduğu klasör

# # Görsellerin yüklenmesi için ImageDataGenerator
# datagen = ImageDataGenerator(rescale=1.0/255)  # Normalize
# dataset = datagen.flow_from_directory(
#     data_dir,
#     target_size=(160, 160),  # Görsellerin boyutlandırılması
#     batch_size=32,
#     class_mode='sparse',  # Etiketlerin sayısal bir sınıf tipi olması için
#     shuffle=False  # Sınıf ve dizin sırasını korumak için
# )

# # GetTriplets Sınıfını Test Etme
# triplets = GetTriplets(dataset)

# # Bir batch üçlü örneklerin yazdırılması
# for batch_index in range(1):  # Test için ilk batch
#     # anchor_imgs, positive_imgs, negative_imgs = triplets[batch_index]
    
#     # Birinci batch'in üçlülerini almak ve dosya yollarını yazdırmak
#     anchor_imgs, positive_imgs, negative_imgs = triplets[0]  # İlk batch'i alıyoruz

#     # Anchor, positive ve negative dosya yollarını yazdırma
#     print("Anchor Görseller (Dosya Yolları):")
#     for i, anchor_path in enumerate(triplets.data[:5]):  # İlk 5 üçlü
#         print(f"Anchor {i + 1}: {anchor_path[0]}")  # İlk eleman anchor'un yolu

#     print("\nPositive Görseller (Dosya Yolları):")
#     for i, positive_path in enumerate(triplets.data[:5]):
#         print(f"Positive {i + 1}: {positive_path[1]}")  # İkinci eleman positive'un yolu

#     print("\nNegative Görseller (Dosya Yolları):")
#     for i, negative_path in enumerate(triplets.data[:5]):
#         print(f"Negative {i + 1}: {negative_path[2]}")  # Üçüncü eleman negative'in yolu