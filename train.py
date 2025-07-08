# Created by Fatma Büşra Tilki / fatmabusratilki
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tqdm import tqdm
import gc
import pandas as pd
from tensorflow.keras.preprocessing import image
import albumentations as A
from tensorflow.keras import layers, models, regularizers
import random #
import os
from facenet import FaceNet
from loss import TripletLoss
from create_triplets import GetTriplets
from early_stopping import EarlyStopping

print("GPU detected:", tf.config.list_physical_devices('GPU'))


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


def train_model(model,
    train_data_dir, # path to train directory
    val_data_dir,   # path to val directory
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
        print(f"Loading pretrained model from {load_model_path}")
        try:
            model.load_weights(load_model_path)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
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
    base_datagen = ImageDataGenerator(rescale=1./255)

    # Instantiate the custom triplet generators
    try:
        train_triplets_generator = GetTriplets(
            directory=train_data_dir,
            image_data_generator=base_datagen, # Pass the base generator
            batch_size=batch_size,
            target_size=(160, 160)
        )
        val_triplets_generator = GetTriplets(
            directory=val_data_dir,
            image_data_generator=base_datagen, # Pass the base generator
            batch_size=batch_size,
            target_size=(160, 160)
        )
        print(f"Training dataset created with {len(train_triplets_generator) * batch_size} potential triplets per epoch.")
        print(f"Validation dataset created with {len(val_triplets_generator) * batch_size} potential triplets per epoch.")

    except ValueError as e:
        print(f"Error creating Triplet Dataset: {e}")
        print("Please check your dataset directories and ensure classes have enough images.")
        return None # Stop training if dataset creation fails


    loss_fn = TripletLoss(alpha=margin) # Use the margin parameter

    # === Cosine Decay Scheduler ===
    # Calculate decay steps based on the length of the custom generator
    # Note: The number of triplets per epoch can vary based on generation strategy.
    # Using the generator's __len__ gives the number of *batches*.
    steps_per_epoch = len(train_triplets_generator)
    decay_steps = epochs * steps_per_epoch

    # Ensure decay_steps is at least 1 to avoid errors if dataset is empty or very small
    decay_steps = max(1, decay_steps)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        alpha=0.0  # Min LR = 0
    )

    # === Optimizer (with LR Schedule) ===
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=lr_schedule)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr_schedule, rho=0.9)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate=learning_rate, initial_accumulator_value=0.1)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}. Choose from 'adam', 'sgd', 'rmsprop', or 'adagrad'.")

    # === Compile the model ===
    # When using a custom training loop, model.compile is not strictly necessary for optimizer/loss,
    # but it can be useful for metrics. However, the triplet_accuracy metric requires triplet inputs.
    # Let's compute metrics manually in the loop for clarity.
    # model.compile(optimizer=optimizer, loss=loss_fn) # Cannot use model.compile easily with triplet metric


    # === Early Stopping ===
    early_stopping = EarlyStopping(patience=5, delta=0.0001) # Use default delta or configure
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
    print("\nTriplet loss Training started...")


    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        loss_fn.alpha = final_margin + (initial_margin - final_margin) * (1 + np.cos(np.pi * epoch / total_epochs)) / 2

        print(f"\nEpoch {epoch+1}/{epochs} | Margin: {loss_fn.alpha:.2f}")

        # Show current LR
        current_step = epoch * steps_per_epoch
        current_lr = lr_schedule(tf.cast(current_step, tf.float32)).numpy() # Use the schedule to get LR
        print(f"Current Learning Rate: {current_lr:.8f}")

        # === Training ===
        running_loss = 0.0
        running_acc = 0.0 # Manually track accuracy
        train_batches = len(train_triplets_generator)

        if train_batches == 0:
             print("Skipping training epoch, train_triplets_generator yielded 0 batches.")
             train_loss = 0
             train_acc = 0
        else:
            for step in tqdm(range(train_batches), desc=f"Train {epoch + 1}/{epochs}"):
                (anchors, positives, negatives), _ = train_triplets_generator[step] # Get batch from Sequence

                # Move data to device (if using a multi-device strategy, Keras handles this)
                # For single GPU/CPU, TF handles device placement automatically by default

                with tf.GradientTape() as tape:
                    emb_a = model(anchors, training=True)
                    emb_p = model(positives, training=True)
                    emb_n = model(negatives, training=True)

                    # Compute loss
                    loss = loss_fn.call(None, (emb_a, emb_p, emb_n)) # Call the loss function
                    l2_penalty = 1e-4 * sum(tf.nn.l2_loss(var) for var in model.trainable_variables)
                    loss += l2_penalty  # Weight decay ekle

                    # Compute accuracy for this batch
                    acc = triplet_accuracy(None, [emb_a, emb_p, emb_n], margin=loss_fn.alpha) # Use the current margin


                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                running_loss += loss.numpy()
                running_acc += acc.numpy() # Accumulate accuracy

            train_loss = running_loss / train_batches
            train_acc = running_acc / train_batches


        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        # === Validation ===
        val_loss_total = 0.0
        val_acc_total = 0.0 # Manually track accuracy
        val_batches = len(val_triplets_generator)

        if val_batches == 0:
            print("Skipping validation epoch, val_triplets_generator yielded 0 batches.")
            val_loss = 0
            val_acc = 0
        else:
            # Ensure generator is reset for a full pass
            # val_triplets_generator.on_epoch_end() # Not needed for validation set

            for step in tqdm(range(val_batches), desc=f"Val {epoch + 1}/{epochs}"):
                (anchors, positives, negatives), _ = val_triplets_generator[step] # Get batch

                # No gradient tape needed for validation
                emb_a = model(anchors, training=False)
                emb_p = model(positives, training=False)
                emb_n = model(negatives, training=False)

                # Compute loss
                loss = loss_fn.call(None, (emb_a, emb_p, emb_n)) # Call the loss function

                # Compute accuracy for this batch
                acc = triplet_accuracy(None, [emb_a, emb_p, emb_n], margin=loss_fn.alpha) # Use the current margin


                val_loss_total += loss.numpy()
                val_acc_total += acc.numpy() # Accumulate accuracy

            val_loss = val_loss_total / val_batches
            val_acc = val_acc_total / val_batches

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # === Save best model ===
        # Save based on validation loss (common) or validation accuracy
        # Since the goal is distance-based accuracy, saving based on val_acc might be better
        # Let's save based on val_acc.
        if val_loss < best_val_loss: # Note: Comparison was val_loss < best_val_loss, changed to val_acc > best_val_acc
            best_val_loss = val_loss # Store best accuracy now
            try:
                model.save(best_model_path, include_optimizer=False, save_format='keras') # Save weights only or full model without optimizer state
                print(f"Best model saved at epoch {epoch + 1} with val_loss: {val_loss:.4f}")
            except Exception as e:
                print(f"Error saving best model: {e}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        # Regenerate triplet pool at the end of the epoch for variety
        train_triplets_generator.on_epoch_end()


    # === Save final model ===
    try:
        model.save(save_path, include_optimizer=False, save_format='keras') # Save weights only or full model without optimizer state
        print(f"Final model saved to {save_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")

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
    plt.figure(figsize=(16, 6))
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
    train_data = "celeba_balanced/train"
    val_data = "celeba_balanced/val"
    # Model save path
    save_path = "model.keras"
    best_model_path="best_model.keras"
    model = FaceNet(embedding_size=128)
    batch_size = 16
    epoch = 100
    learning_rate = 1e-3
    optimizer='adagrad'  # 'adam', 'sgd', 'rmsprop', 'adagrad'
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
        load_model_path=None,
        freeze_base=False
    )

    plot_history(history)
    print("Training completed and history plotted.")
