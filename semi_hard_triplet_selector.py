# Created by Fatma Büşra Tilki / fatmabusratilki
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence


class SemiHardTripletSelector(Sequence):
    def __init__(self, model, directory, batch_size=32, target_size=(160, 160), margin=0.2, preprocess_fn=None, max_triplets_per_epoch=500):
        """
        model: Embedding model (Keras).
        directory: Data directory (same structure as flow_from_directory).
        batch_size: Triplet batch size.
        target_size: Target size for images.
        margin: Initial margin, updated dynamically.
        preprocess_fn: Optional preprocessing (e.g. rescaling).
        max_triplets_per_epoch: Max number of triplets to select to reduce memory use.
        """
        self.model = model
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.margin = margin
        self.preprocess_fn = preprocess_fn
        self.max_triplets = max_triplets_per_epoch

        self.class_to_paths = {}
        self.labels = []
        self.paths = []
        self.label_to_index = {}
        self.index_to_label = {}

        self._load_paths()
        self._prepare_epoch()

    def _load_paths(self):
        class_names = sorted([d for d in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, d))])
        self.label_to_index = {cls: i for i, cls in enumerate(class_names)}
        self.index_to_label = {i: cls for cls, i in self.label_to_index.items()}

        for cls in class_names:
            class_dir = os.path.join(self.directory, cls)
            image_paths = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)
                           if fname.lower().endswith((".jpg", ".jpeg", ".png"))]
            if len(image_paths) >= 2:  # Require at least two images per class
                label_idx = self.label_to_index[cls]
                self.class_to_paths[label_idx] = image_paths
                self.paths.extend(image_paths)
                self.labels.extend([label_idx] * len(image_paths))

        self.labels = np.array(self.labels)
        self.paths = np.array(self.paths)

    def _prepare_epoch(self):
        """Generates embeddings and finds semi-hard triplets for the entire dataset."""
        # Load all images and embed them

        X = np.array([self._load_and_preprocess_image(p) for p in self.paths])
        embeddings = self.model.predict(X, batch_size=64, verbose=0)

        # Compute pairwise distances
        distances = pairwise_distances(embeddings, metric="euclidean")

        # Generate triplets
        self.triplets = self._mine_semi_hard_triplets(embeddings, distances)
        random.shuffle(self.triplets)

        # Limit the number of triplets to control memory
        self.triplets = self.triplets[:self.max_triplets]
 

    def _mine_semi_hard_triplets(self, embeddings, distances):
        """Returns list of (anchor_idx, pos_idx, neg_idx) satisfying semi-hard conditions."""
        triplets = []
        for anchor_idx in range(len(embeddings)):
            anchor_label = self.labels[anchor_idx]

            # Positive samples (same class, different image)
            pos_indices = np.where(self.labels == anchor_label)[0]
            pos_indices = pos_indices[pos_indices != anchor_idx]

            if len(pos_indices) == 0:
                continue

            for pos_idx in pos_indices:
                d_ap = distances[anchor_idx, pos_idx]

                # Negative samples (different class)
                neg_indices = np.where(self.labels != anchor_label)[0]
                semi_hard_negatives = [neg_idx for neg_idx in neg_indices
                                       if d_ap < distances[anchor_idx, neg_idx] < d_ap + self.margin]

                if semi_hard_negatives:
                    neg_idx = random.choice(semi_hard_negatives)
                    triplets.append((anchor_idx, pos_idx, neg_idx))
        return triplets

    def _load_and_preprocess_image(self, path):
        img = image.load_img(path, target_size=self.target_size)
        img = image.img_to_array(img)
        
        if self.preprocess_fn:
            img = self.preprocess_fn(img)
        else:
            img = img / 255.0

        img = augmentation_gen.random_transform(img)

        return img

    def __len__(self):
        return len(self.triplets) // self.batch_size

    def __getitem__(self, idx):
        batch_triplets = self.triplets[idx * self.batch_size:(idx + 1) * self.batch_size]
        anchor_imgs, pos_imgs, neg_imgs = [], [], []

        for a_idx, p_idx, n_idx in batch_triplets:
            anchor_imgs.append(self._load_and_preprocess_image(self.paths[a_idx]))
            pos_imgs.append(self._load_and_preprocess_image(self.paths[p_idx]))
            neg_imgs.append(self._load_and_preprocess_image(self.paths[n_idx]))

        return [np.array(anchor_imgs), np.array(pos_imgs), np.array(neg_imgs)], None

    def on_epoch_end(self):
        self._prepare_epoch()
