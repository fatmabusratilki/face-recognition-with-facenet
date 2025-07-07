# file ipython-input-13-db064894ced8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tqdm import tqdm
import gc
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models, regularizers
import random 
import os
from tensorflow.keras.utils import Sequence
from sklearn.metrics.pairwise import pairwise_distances
from hard_triplet_generator import TripletFolderLoader

print("GPU detected:", tf.config.list_physical_devices('GPU'))

augmentation_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # zoom_range=0.3,
    # shear_range=0.2, 
    # brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)

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
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(embedding_size, activation=None)(x)

    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)
    return models.Model(inputs, outputs)

class TripletLoss(tf.keras.losses.Loss): # Keras Loss sınıfından miras alması daha doğru
    def __init__(self, alpha=0.2, name="TripletLoss"):
        super().__init__(name=name)
        """
        Initialize the Triplet Loss.

        Parameters:
        - alpha: The margin between positive and negative pairs.
        """
        self.alpha = alpha

    def call(self, y_true, y_pred): # Keras Loss standard call signature
        """
        Compute the triplet loss. Assumes y_pred is a tuple/list of (anchor, positive, negative) embeddings.
        y_true is ignored, but required by the Keras API.

        Parameters:
        - y_true: Ignored.
        - y_pred: A tuple or list containing (anchor_embeddings, positive_embeddings, negative_embeddings).

        Returns:
        - The triplet loss value.
        """
        anchor, positive, negative = y_pred

        # Compute the distance between anchor and positive
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)

        # Compute the distance between anchor and negative
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

        # Compute the triplett loss
        loss = tf.maximum(pos_dist - neg_dist + self.alpha, 0.0)

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
        })
        return config

# Calculate triplet accuracy
def triplet_accuracy(y_true, y_pred, margin = 0.2):
    # This metric function will work during model.fit
    # It expects y_pred to be the model's output (embeddings)
    # and y_true is usually ignored for triplet loss accuracy
    # We need to find a way to pass the anchors, positives, negatives to the metric
    # A common approach is to create a custom training loop or a custom Model class

    # As a simple metric for validation/manual checks, assuming y_pred is [anchor, positive, negative] embeddings:
    if isinstance(y_pred, (list, tuple)) and len(y_pred) == 3:
         anchor, positive, negative = y_pred
    else:
        # If y_pred is just the raw model output, this metric won't work directly in model.compile/fit
        # Placeholder or error handling
        # print("Warning: triplet_accuracy metric is not receiving the expected triplet embeddings.")
        return 0.0 # Or raise an error


    # Euclidean distance between anchor and positive, anchor and negative
    positive_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    negative_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    # Check if positive distance is less than negative distance
    correct = tf.cast(positive_distance + margin < negative_distance, tf.float32)

    # Accuracy = correct predictions / total predictions
    accuracy = tf.reduce_mean(correct)

    return accuracy


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
                                       if d_ap < distances[anchor_idx, neg_idx] < d_ap + self.margin] # burda margini 0.4 olarak ayarla 

                if semi_hard_negatives:
                    neg_idx = random.choice(semi_hard_negatives)
                    triplets.append((anchor_idx, pos_idx, neg_idx))
        return triplets

    # def _load_and_preprocess_image(self, path):
    #     img = image.load_img(path, target_size=self.target_size)
    #     img = image.img_to_array(img)
    #     if self.preprocess_fn:
    #         img = self.preprocess_fn(img)
    #     else:
    #         img = img / 255.0
    #     return img

    # SemiHardTripletSelector içinde:
    def _load_and_preprocess_image(self, path):
        img = image.load_img(path, target_size=self.target_size)
        img = image.img_to_array(img)
        
        if self.preprocess_fn:
            img = self.preprocess_fn(img)
        else:
            img = img / 255.0

        # ImageDataGenerator’dan rastgele augmentation uygulamak için:
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

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



def train_model(model,
    train_data_dir, # Renamed for clarity: path to train directory
    val_data_dir,   # Renamed for clarity: path to val directory
    batch_size=32,
    epochs=10,
    learning_rate=3e-4,
    save_path="model.keras",
    best_model_path="best_model.keras",
    optimizer='adam',
    margin=0.2, # Triplet Loss margin
    load_model_path=None,
    freeze_base=False
):

    # === Load pretrained weights if specified ===
    if load_model_path is not None:
        print(f"📦 Loading pretrained model from {load_model_path}")
        try:
            model.load_weights(load_model_path)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}")
            print("Training will start without pretrained weights.")
            load_model_path = None # Reset to None if loading fails


    if freeze_base:
        print("🧊 Freezing base model layers...")
        # Identify base layers to freeze. This depends on your model structure.
        # Assuming the first few layers/blocks are the base.
        # You might need to inspect your specific FaceNet model definition.
        # For this general FaceNet, let's freeze conv and initial layers.
        for layer in model.layers:
             # Example: Freeze layers by name or type
             if isinstance(layer, (layers.Conv2D, layers.BatchNormalization, layers.MaxPooling2D)):
                 layer.trainable = False
             elif layer.name in ['FaceNet_input', 'lambda']: # Don't freeze input or output normalization layer
                 layer.trainable = True
             else: # You might need to adjust this based on your model's structure
                 layer.trainable = False # Freeze other layers by default
        print("Base layers frozen. Check model.summary() to confirm.")
        # model.summary() # Uncomment to see which layers are trainable


    # === Data Generators ===
    # Use a base ImageDataGenerator for common preprocessing (like rescaling)
    # Triplet generation logic is handled by the GetTriplets Sequence.
    # base_datagen = ImageDataGenerator(rescale=1./255)

    # train_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     rotation_range=15,  
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     zoom_range=0.3,
    #     shear_range=0.2, 
    #     brightness_range=[0.5, 1.5],
    #     fill_mode='nearest'
    # )

    base_datagen_preprocess = lambda x: x / 255.0

     # --- Initialize SemiHardTripletSelector for train and val ---
    print("🎯 Creating training triplet selector...")
    train_triplets_selector = SemiHardTripletSelector(
        model=model,
        directory=train_data_dir,
        batch_size=batch_size,
        target_size=(160, 160),
        margin=margin,
        preprocess_fn=base_datagen_preprocess,
        max_triplets_per_epoch=2048
    )

    print("🎯 Creating validation triplet selector...")
    val_triplets_selector = SemiHardTripletSelector(
        model=model,
        directory=val_data_dir,
        batch_size=batch_size,
        target_size=(160, 160),
        margin=margin,
        preprocess_fn=base_datagen_preprocess,
        max_triplets_per_epoch=512
    )
    if len(train_triplets_selector) == 0 or len(val_triplets_selector) == 0:
        print("🚫 Dataset veya triplet havuzunda sorun var. Lütfen verinizi kontrol edin.")
        return None
    
    # train_triplets_selector._load_and_preprocess_image = lambda path: augmentation_gen.random_transform(
    #     image.img_to_array(image.load_img(path, target_size=(160,160))) / 255.0
    # )

    loss_fn = TripletLoss(alpha=margin) # Use the margin parameter

    # === Cosine Decay Scheduler ===
    steps_per_epoch = len(train_triplets_selector)
    decay_steps = max(1, epochs * steps_per_epoch)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        alpha=0.0
    )
 

    # === Optimizer (with LR Schedule) ===
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=lr_schedule)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr_schedule, rho=0.9)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate=lr_schedule, initial_accumulator_value=0.1)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}. Choose from 'adam', 'sgd', 'rmsprop', or 'adagrad'.")

 


    # === Early Stopping ===
    early_stopping = EarlyStopping(patience=10, delta=0.001) # Use default delta or configure
    best_val_loss = float("inf") # Değişiklik: En iyi doğruluk için takip ediyoruz

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [], # Accuracy computed manually
        "val_accuracy": []    # Accuracy computed manually
    }

    initial_margin = margin # Store initial margin for cosine decay
    final_margin = 0.2
    total_epochs = epochs

    # === Triplet Loss Training Loop ===
    print("\n🚀 Triplet loss Training started...")


    for epoch in range(epochs):
        print(f"\n🔁 Epoch {epoch+1}/{epochs}")

        loss_fn.alpha = final_margin + (initial_margin - final_margin) * (1 + np.cos(np.pi * epoch / total_epochs)) / 2

        print(f"\n🔁 Epoch {epoch+1}/{epochs} | Margin: {loss_fn.alpha:.4f}")

        # Update margin in selectors & refresh triplets
        train_triplets_selector.margin = loss_fn.alpha
        train_triplets_selector.on_epoch_end()

        val_triplets_selector.margin = loss_fn.alpha
        val_triplets_selector.on_epoch_end()

        current_step = epoch * steps_per_epoch
        current_lr = lr_schedule(tf.cast(current_step, tf.float32)).numpy()
        print(f"Learning Rate: {current_lr:.8f}")

        # === Training ===
        running_loss = 0.0
        running_acc = 0.0 # Manually track accuracy

        train_batches = len(train_triplets_selector)
        
        for step in tqdm(range(train_batches), desc=f"Train {epoch + 1}/{epochs}"):
            (anchors, positives, negatives), _ = train_triplets_selector[step]
            
            with tf.GradientTape() as tape:
                emb_a = model(anchors, training=True)
                emb_p = model(positives, training=True)
                emb_n = model(negatives, training=True)

                loss = loss_fn(None, (emb_a, emb_p, emb_n))
                l2_penalty = 1e-4 * sum(tf.nn.l2_loss(var) for var in model.trainable_variables)
                loss += l2_penalty

                acc = triplet_accuracy(None, [emb_a, emb_p, emb_n], margin=loss_fn.alpha)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            running_loss += loss.numpy()
            running_acc += acc.numpy()

        train_loss = running_loss / train_batches
        train_acc = running_acc / train_batches
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

    
        # === Validation ===
        val_loss_total = 0.0
        val_acc_total = 0.0 # Manually track accuracy
        val_batches = len(val_triplets_selector)

     
        for step in tqdm(range(val_batches), desc=f"Val {epoch + 1}/{epochs}"):
            (anchors, positives, negatives), _ = val_triplets_selector[step]
            emb_a = model(anchors, training=False)
            emb_p = model(positives, training=False)
            emb_n = model(negatives, training=False)

            loss = loss_fn(None, (emb_a, emb_p, emb_n))
            acc = triplet_accuracy(None, [emb_a, emb_p, emb_n], margin=loss_fn.alpha)

            val_loss_total += loss.numpy()
            val_acc_total += acc.numpy()

        val_loss = val_loss_total / val_batches
        val_acc = val_acc_total / val_batches
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"📊 Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # === Save best model ===
        # Save based on validation loss (common) or validation accuracy
        # Since the goal is distance-based accuracy, saving based on val_acc might be better
        # Let's save based on val_acc.
        if val_loss < best_val_loss: # Note: Comparison was val_loss < best_val_loss, changed to val_acc > best_val_acc
            best_val_loss = val_loss # Store best accuracy now
            try:
                model.save(best_model_path, include_optimizer=False, save_format='keras') # Save weights only or full model without optimizer state
                print(f"✅ Best model saved at epoch {epoch + 1} with val_acc: {val_loss:.4f}")
            except Exception as e:
                print(f"⚠️ Error saving best model: {e}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


    # === Save final model ===
    try:
        model.save(save_path, include_optimizer=False, save_format='keras') # Save weights only or full model without optimizer state
        print(f"✅ Final model saved to {save_path}")
    except Exception as e:
        print(f"⚠️ Error saving final model: {e}")

    tf.keras.backend.clear_session()
    gc.collect()

    return history

def plot_history(history, save_path_prefix="training_history"):
    """
    Plot training and validation loss and accuracy in two separate plots.
    """
    if not history:
        print("No history data to plot.")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    plt.figure(figsize=(16, 12))
    plt.subplot(1, 2, 1)  # Two rows, one column, first subplot
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
  
  
    # Plot Accuracy
    plt.subplot(1, 2, 2)  # Two rows, one column, second subplot
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
  
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}.png")
    plt.close()
    print(f"Training history plots saved to {save_path_prefix}.png")


if __name__ == "__main__":

    # Data paths
    train_data = "celeba_balanced/val"  # Adjusted to match your dataset structure
    val_data = "dataset/processed/test"  # Adjusted to match your dataset structure
    # Model save path
    save_path = "last_model_finetuned.keras"
    best_model_path="last_model_finetuned_best_model.keras"
    model = FaceNet(embedding_size=256)
    batch_size = 16
    epoch = 20
    learning_rate = 1e-5
    optimizer='rmsprop'  # 'adam', 'sgd', 'rmsprop', 'adagrad'
    margin = 0.2

    history = train_model(
        model=model,
        train_data_dir=train_data,
        val_data_dir=val_data,
        batch_size=batch_size,
        epochs=epoch,
        learning_rate=learning_rate,
        save_path=save_path,
        best_model_path=best_model_path,
        optimizer=optimizer,
        margin=margin,
        load_model_path="fine_tune_best_model_same_data.keras",
        freeze_base=False
    )

    plot_history(history)
    print("Training completed and history plotted.")
