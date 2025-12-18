import numpy as np
import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def augment_landmarks(landmarks, num_augmentations=3):
    augmented = [landmarks.flatten()]

    
    for _ in range(num_augmentations):
        aug = landmarks.copy().reshape(21, 3)
        
        angle = np.random.uniform(-30, 30) * np.pi / 180
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        aug = aug @ rotation_matrix.T
        
        scale = np.random.uniform(0.85, 1.15)
        aug = aug * scale
        
        noise = np.random.normal(0, 0.015, aug.shape)
        aug = aug + noise
        
        if np.random.random() > 0.5:
            aug[:, 0] = -aug[:, 0]
        
        augmented.append(aug.flatten())
    
    return np.array(augmented)

def create_model(input_shape, num_classes):
  
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Conv1D(256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        
        GlobalMaxPooling1D(),
        
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history, output_path):
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/training_history.png', dpi=150)
    print(f"Training history plot saved to {output_path}/training_history.png")
    plt.close()

def train():
    DATA_PATH = 'processed_data'
    MODEL_PATH = 'models'
    os.makedirs(MODEL_PATH, exist_ok=True)
    

    print("Training Started")
    
    print("Loading data")
    try:
        X = np.load(f'{DATA_PATH}/X_landmarks.npy')
        y = np.load(f'{DATA_PATH}/y_labels.npy')
        with open(f'{DATA_PATH}/label_map.json', 'r') as f:
            label_map = json.load(f)
    except FileNotFoundError:
        print("ERROR: NO DATA FOUND")
        return
    
    print(f"Loaded {len(X)} samples")
    print(f"Number of classes: {len(label_map)}")
    print(f"Feature shape: {X.shape}")

    print("Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   Class {cls} ({label_map[str(cls)]}): {count} samples")
    
    X = X.reshape(-1, 21, 3)
    
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.50, random_state=42, stratify=y_train
    )

    
    print(" Applying data augmentation")
    X_augmented = []
    y_augmented = []
    
    for landmarks, label in zip(X_train, y_train):
        aug_samples = augment_landmarks(landmarks)
        X_augmented.extend(aug_samples)
        y_augmented.extend([label] * len(aug_samples))
    
    X_train_aug = np.array(X_augmented).reshape(-1, 21, 3)
    y_train_aug = np.array(y_augmented)
    
    X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=42)
    
    print(f"Augmented training samples: {len(X_train_aug)} (original: {len(X_train)})")
    
    print("Building model")
    num_classes = len(label_map)
    model = create_model(input_shape=(21, 3), num_classes=num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'{MODEL_PATH}/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]


    print("Starting training")    
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    plot_training_history(history, MODEL_PATH)
    
    print("Evaluating on test set")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    

    print("TRAINING COMPLETE!")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}/best_model.keras")

    
    model.save(f'{MODEL_PATH}/final_model.keras')
    print(f"Final model also saved to: {MODEL_PATH}/final_model.keras")
    
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    gap = train_acc - val_acc
    
    print("OVERFITTING CHECK:")
    if gap < 0.05:
        print("  Model is generalizing well.")
    elif gap < 0.10:
        print(" Consider more augmentation or regularization.")
    else:
        print("Model might be overfitting")

    print(f"Accuracy gap: {gap*100:.2f}%")

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    train()