import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow import keras
from tqdm import tqdm
from facenet import FaceNet
from loss import TripletLoss
from create_triplets import GetTriplets
from early_stopping import EarlyStopping
import tensorflow as tf
import math
import random
import json


def random_search_experiments(
    train_data, val_data,
    n_trials=10,
    batch_size_choices=[16, 32, 64],
    learning_rate_range=(1e-5, 1e-3),
    embedding_size_choices=[64, 128, 256],
    optimizers=['adam', 'sgd', 'rmsprop', 'adagrad'],
    save_results_path="random_experiment_results.json"
):
    results = []

    for trial in range(n_trials):
        print(f"\n🔁 Running trial {trial + 1}/{n_trials}...")

        batch_size = random.choice(batch_size_choices)
        learning_rate = 10 ** random.uniform(
            math.log10(learning_rate_range[0]),
            math.log10(learning_rate_range[1])
        )
        embedding_size = random.choice(embedding_size_choices)
        optimizer = random.choice(optimizers)

        model = FaceNet(embedding_size=embedding_size)
        save_path = f"models/random_model_trial_{trial + 1}.h5"

        history = train_model(
            model,
            train_data=train_data,
            val_data=val_data,
            batch_size=batch_size,
            epochs=10,
            learning_rate=learning_rate,
            save_path=save_path,
            optimizer=optimizer
        )

        result = {
            "trial": trial + 1,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "embedding_size": embedding_size,
            "optimizer": optimizer,
            "final_val_loss": history["val_loss"][-1],
            "final_val_acc": history["val_accuracy"][-1]
        }
        results.append(result)

        # Her trial için grafik kaydet (opsiyonel)
        plot_history(history, save_path=f"training_history_trial_{trial + 1}.png")

    # Sonuçları JSON olarak kaydet
    with open(save_results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Random search completed. Results saved to {save_results_path}")



print("GPU detected:", tf.config.list_physical_devices('GPU'))

def triplet_accuracy(y_true, y_pred):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Euclidean distance between anchor and positive, anchor and negative
    positive_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    negative_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    
    # Doğru eşleşmeleri kontrol et
    correct = tf.cast(positive_distance < negative_distance, tf.float32)
    
    # Accuracy = doğru eşleşmelerin sayısı / toplam triplet sayısı
    accuracy = tf.reduce_mean(correct)
    
    return accuracy


def train_model(model, train_data, val_data, batch_size=16, epochs=10, learning_rate=0.001, save_path="model.h5", optimizer='adam'):
    """
    Train the model with the given parameters.

    Parameters:
    - model: The model to be trained.
    - train_data: The training data.
    - val_data: The validation data.
    - epochs: Number of epochs to train the model.
    - batch_size: Size of the batches of data.
    - learning_rate: Learning rate for the optimizer.
    - save_path: Path to save the trained model.
    - optimizer: Optimizer to use for training.

    Returns:
    - history: Training history object containing training and validation metrics.
    """
    
    # train_datagen = ImageDataGenerator(rescale=1./255,)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Daha geniş dönüş açısı
        # width_shift_range=0.3,
        # height_shift_range=0.3,
        # zoom_range=0.3,
        # shear_range=0.2, 
        # brightness_range=[0.7, 1.3],  # Parlaklık değişimi eklendi
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_dataset = train_datagen.flow_from_directory(
        train_data,
        target_size=(160, 160),
        batch_size=batch_size,
        class_mode='binary'
    )

    val_dataset = val_datagen.flow_from_directory(
        val_data,
        target_size=(160, 160),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Create triplets
    train_triplets = GetTriplets(train_dataset, batch_size=batch_size)
    val_triplets = GetTriplets(val_dataset, batch_size=batch_size)

    # Triplet loss
    loss_fn = TripletLoss(alpha=0.3)

    # Choose optimizer
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
    elif optimizer == 'sgd':    
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate, rho=0.9)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1)
    else:
        raise ValueError("Unsupported optimizer. Choose from 'adam', 'sgd', 'rmsprop', or 'adagrad'.")
    
    print(f'optimizer: {optimizer}')


    # Schedule learning rate
    # scheduler = LearningRateScheduler(lambda epoch: learning_rate * (0.1 ** (epoch // 2)), verbose=1)

    # Early stopping
    early_stopping = EarlyStopping()

    history = {
        "train_loss": [],
        "val_loss": [], 
        "train_accuracy": [],
        "val_accuracy": []
    }

    # Training loop

    for epoch in range(epochs):
        running_loss = 0.0
        epoch_accuracy = 0.0
        train_true, train_pred = [], []
        val_true, val_pred = [], []

        for anchors, positives, negatives in tqdm(train_triplets, desc=f"Epoch {epoch + 1}/{epochs}"):
            with tf.GradientTape() as tape:
                anchor_embeddings = model(anchors, training=True)
                positive_embeddings = model(positives, training=True)
                negative_embeddings = model(negatives, training=True)

                loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                accuracy = triplet_accuracy(None, [anchor_embeddings, positive_embeddings, negative_embeddings])

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            running_loss += loss.numpy()
            epoch_accuracy += accuracy.numpy()
        
            # train_true.extend([1] * len(anchors))
            # train_pred.extend([1 if np.mean(positive_embeddings[i]) < threshold else 0 for i in range(len(anchor_embeddings))])


        train_loss = running_loss / len(train_triplets)
        train_accuracy = epoch_accuracy / len(train_triplets)
        # train_accuracy = accuracy_score(train_true, train_pred)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)


        # Validation loop
        val_running_loss = 0.0
        val_epoch_accuracy = 0.0
        val_true, val_pred = [], []

        for anchors, positives, negatives in tqdm(val_triplets, desc=f"Validation {epoch + 1}/{epochs}"):
                
            anchor_embeddings = model(anchors, training=False)
            positive_embeddings = model(positives, training=False)
            negative_embeddings = model(negatives, training=False)
                
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            accuracy_val = triplet_accuracy(None, [anchor_embeddings, positive_embeddings, negative_embeddings])

            val_running_loss += loss.numpy()
            val_epoch_accuracy += accuracy_val.numpy()


            # val_true.extend([1] * len(anchors))
            # val_pred.extend([1 if np.mean(positive_embeddings[i]) < threshold else 0 for i in range(len(anchor_embeddings))])

        val_loss = val_running_loss / len(val_triplets)
        val_accuracy = val_epoch_accuracy / len(val_triplets)
        # val_accuracy = accuracy_score(val_true, val_pred)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Save the model
    model.save(save_path, include_optimizer=True, save_format='h5')
    print(f"Model saved to {save_path}")

    return history

def plot_history(history, save_path="training_history.png"):
    """
    Plot training and validation loss and accuracy.
    """
    plt.figure(figsize=(10,6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Training and Validation Metrics')
    plt.savefig(save_path)
    #plt.show()

if __name__ == "__main__":

    # Data paths
    train_data = "datasets/train"
    val_data = "datasets/val"
    # train_data = "CelebA/processed/train"
    # val_data = "CelebA/processed/val"
    # train_data = "celeba_balanced/train"
    # val_data = "celeba_balanced/val"
    save_path = "model.h5"

    # Model parameters
    model = FaceNet(embedding_size=256)  # 128, 256, 512
    # model.compile(optimizer='adam', loss=TripletLoss(alpha=0.3), metrics=[triplet_accuracy])
    batch_size = 64
    epochs = 8
    learning_rate = 7.005382843891185e-05
    optimizer = 'adam'  # Choose from 'adam', 'sgd', 'rmsprop', or 'adagrad'
    # optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    threshold = 0.4

    # Train the model
    history = train_model(
        model, 
        train_data, 
        val_data, 
        batch_size=batch_size, 
        epochs=epochs, 
        learning_rate=learning_rate, 
        save_path=save_path, 
        optimizer=optimizer
    )

    plot_history(history, save_path="training_history.png")
    print("Training completed and history plotted.")

 
    # random_search_experiments(
    #     train_data="celeba_balanced/train",
    #     val_data="celeba_balanced/val",
    #     n_trials=20  # veya 20
    # )
