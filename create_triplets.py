# Created by Fatma Büşra Tilki / fatmabusratilki
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class GetTriplets(tf.keras.utils.Sequence):
    def __init__(self, directory, image_data_generator, batch_size=32, target_size=(160, 160)):
        """
        directory: Root directory of the dataset (e.g., train, val).
        image_data_generator: An instance of tf.keras.preprocessing.image.ImageDataGenerator.
        batch_size: Batch size for triplets.
        target_size: Target size for images.
        """
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.batch_size = batch_size
        self.target_size = target_size

        self.class_to_paths = {}
        self.classes = sorted(os.listdir(directory)) # Get class names (folder names)
        self.num_classes = len(self.classes)

        if self.num_classes == 0:
             raise ValueError(f"No classes found in the directory: {directory}")


        all_paths = []
        all_labels = []
        self.label_to_class_name = {i: class_name for i, class_name in enumerate(self.classes)}
        self.class_name_to_label = {class_name: i for i, class_name in enumerate(self.classes)}


        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(image_files) > 0:
                    self.class_to_paths[i] = image_files
                    all_paths.extend(image_files)
                    all_labels.extend([i] * len(image_files))

        self.all_paths = all_paths
        self.all_labels = all_labels
        self.num_images = len(self.all_paths)

        if self.num_images == 0:
             raise ValueError(f"No images found in the directory: {directory}")

        self.labels = list(self.class_to_paths.keys()) # List of labels with at least one image
        self.labels_with_min_two_images = [label for label, paths in self.class_to_paths.items() if len(paths) >= 2]

        if not self.labels_with_min_two_images:
             raise ValueError(f"No classes with at least two images found in {directory} to form positive pairs.")


        # Pre-generate a pool of triplet indices for one epoch
        self._triplet_indices = self._generate_triplet_indices_pool()


    def _generate_triplet_indices_pool(self, pool_size_multiplier=1):
        """Generates a large pool of triplet indices for one epoch."""
        pool_size = self.num_images * pool_size_multiplier # Generate more triplets than images to sample from
        triplet_indices = []

        # We need indices into self.all_paths/self.all_labels
        # Let's create a mapping from path to index in all_paths
        path_to_index = {path: i for i, path in enumerate(self.all_paths)}


        # Generate triplets based on labels
        temp_triplets = []
        # Iterate through each class that has at least 2 images
        for anchor_label in self.labels_with_min_two_images:
            anchor_positive_paths = self.class_to_paths[anchor_label]

            if len(anchor_positive_paths) < 2:
                continue

            # Generate triplets where the anchor is from this class
            for _ in range(pool_size // len(self.labels_with_min_two_images)): # Aim for roughly equal distribution across anchor classes
                 # Choose anchor and positive from the same class
                a_path, p_path = random.sample(anchor_positive_paths, 2)

                # Choose a negative class different from the anchor class
                negative_label = random.choice([l for l in self.labels if l != anchor_label])
                # Choose a negative image from the negative class
                n_path = random.choice(self.class_to_paths[negative_label])

                # Get indices from path_to_index
                a_idx = path_to_index[a_path]
                p_idx = path_to_index[p_path]
                n_idx = path_to_index[n_path]

                temp_triplets.append((a_idx, p_idx, n_idx))

        # Shuffle the generated triplets pool
        random.shuffle(temp_triplets)
        return temp_triplets


    def __len__(self):
        # The number of batches per epoch is the total number of generated triplets divided by batch size
        return len(self._triplet_indices) // self.batch_size

    def __getitem__(self, index):
        """Generates one batch of data."""
        # Get a batch of triplet indices from the pre-generated pool
        batch_indices = self._triplet_indices[index * self.batch_size : (index + 1) * self.batch_size]

        batch_anchor_imgs, batch_positive_imgs, batch_negative_imgs = [], [], []

        for a_idx, p_idx, n_idx in batch_indices:

            anchor_path = self.all_paths[a_idx]
            positive_path = self.all_paths[p_idx]
            negative_path = self.all_paths[n_idx]

            # Load and transform each image
            anchor_img = self._load_and_transform_image(anchor_path)
            positive_img = self._load_and_transform_image(positive_path)
            negative_img = self._load_and_transform_image(negative_path)

            batch_anchor_imgs.append(anchor_img)
            batch_positive_imgs.append(positive_img)
            batch_negative_imgs.append(negative_img)


        # Convert lists of images to numpy arrays
        batch_anchor_imgs = np.array(batch_anchor_imgs)
        batch_positive_imgs = np.array(batch_positive_imgs)
        batch_negative_imgs = np.array(batch_negative_imgs)

        return [batch_anchor_imgs, batch_positive_imgs, batch_negative_imgs], None # Return None for y_true

    def _load_and_transform_image(self, img_path):
        """Loads an image and applies the generator's transformations."""
        img = image.load_img(img_path, target_size=self.target_size)
        img_array = image.img_to_array(img)

        img_array = img_array / 255.0 # Normalize

        return img_array


    def on_epoch_end(self):
        """Shuffle triplets or regenerate them at the end of each epoch."""
        # Regenerate the pool of triplets for the next epoch to ensure variety
        self._triplet_indices = self._generate_triplet_indices_pool()
        print(f"✅ Regenerated triplet pool for the next epoch. Total triplets: {len(self._triplet_indices)}")
