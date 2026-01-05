"""
Pneumonia Detection Training Pipeline
Supports multiple architectures with transfer learning
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import cv2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class PneumoniaDetector:
    """Main class for pneumonia detection model training and evaluation"""
    
    def __init__(self, data_dir='chest_xray', img_size=(224, 224), batch_size=32):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.model_name = None
        
        # Create outputs directory
        self.output_dir = Path('outputs')
        self.output_dir.mkdir(exist_ok=True)
        
    def prepare_data(self):
        """Prepare data generators with augmentation"""
        print("🔄 Preparing data generators...")
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation and test data (only rescaling)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            self.data_dir / 'val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            self.data_dir / 'test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"✅ Training samples: {self.train_generator.samples}")
        print(f"✅ Validation samples: {self.val_generator.samples}")
        print(f"✅ Test samples: {self.test_generator.samples}")
        print(f"✅ Class indices: {self.train_generator.class_indices}")
        
    def build_model(self, architecture='resnet50', learning_rate=0.0001):
        """Build model with specified architecture"""
        self.model_name = architecture
        print(f"\n🏗️  Building {architecture.upper()} model...")
        
        # Choose base model
        if architecture == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif architecture == 'efficientnetb0':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build complete model
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print(f"✅ Model built successfully!")
        print(f"   Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train(self, epochs=20, fine_tune_epochs=10):
        """Train the model with transfer learning and fine-tuning"""
        print(f"\n🚀 Starting training for {self.model_name}...")
        
        # Create model-specific output directory
        model_dir = self.output_dir / self.model_name
        model_dir.mkdir(exist_ok=True)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                str(model_dir / 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(str(model_dir / 'training_log.csv'))
        ]
        
        # Phase 1: Train with frozen base
        print("\n📊 Phase 1: Training with frozen base model...")
        history1 = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning
        print("\n📊 Phase 2: Fine-tuning with unfrozen layers...")
        
        # Unfreeze the base model
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze early layers, unfreeze later layers
        for layer in base_model.layers[:int(len(base_model.layers) * 0.8)]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        history2 = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            key: history1.history[key] + history2.history[key]
            for key in history1.history.keys()
        }
        
        print(f"\n✅ Training completed!")
        
        # Save final model
        self.model.save(str(model_dir / 'final_model.h5'))
        
        return self.history
    
    def evaluate(self):
        """Evaluate model on test set"""
        print(f"\n📈 Evaluating {self.model_name} on test set...")
        
        model_dir = self.output_dir / self.model_name
        
        # Get predictions
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = self.test_generator.classes
        
        # Calculate metrics
        results = classification_report(
            y_true, y_pred,
            target_names=['NORMAL', 'PNEUMONIA'],
            output_dict=True
        )
        
        print("\n📊 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))
        
        # Save results
        with open(model_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['NORMAL', 'PNEUMONIA'],
                    yticklabels=['NORMAL', 'PNEUMONIA'])
        plt.title(f'{self.model_name.upper()} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(model_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name.upper()} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(model_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Evaluation complete! AUC: {roc_auc:.4f}")
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        model_dir = self.output_dir / self.model_name
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history['loss'], label='Train')
        axes[0, 1].plot(self.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history['precision'], label='Train')
        axes[1, 0].plot(self.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history['recall'], label='Train')
        axes[1, 1].plot(self.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall (Sensitivity)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle(f'{self.model_name.upper()} - Training History', fontsize=16, y=1.00)
        plt.tight_layout()
        plt.savefig(model_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training history saved!")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🏥 PNEUMONIA DETECTION TRAINING PIPELINE")
    print("=" * 60)
    
    # Architectures to train
    architectures = ['resnet50', 'efficientnetb0']
    
    results_summary = {}
    
    for arch in architectures:
        print(f"\n\n{'='*60}")
        print(f"Training {arch.upper()}")
        print(f"{'='*60}")
        
        # Initialize detector
        detector = PneumoniaDetector()
        
        # Prepare data (only once is fine, but we'll do it each time for clarity)
        detector.prepare_data()
        
        # Build model
        detector.build_model(architecture=arch)
        
        # Train
        detector.train(epochs=15, fine_tune_epochs=10)
        
        # Plot history
        detector.plot_training_history()
        
        # Evaluate
        results = detector.evaluate()
        
        # Store summary
        results_summary[arch] = {
            'accuracy': results['accuracy'],
            'precision': results['PNEUMONIA']['precision'],
            'recall': results['PNEUMONIA']['recall'],
            'f1-score': results['PNEUMONIA']['f1-score']
        }
    
    # Print final comparison
    print("\n\n" + "="*60)
    print("📊 FINAL RESULTS COMPARISON")
    print("="*60)
    
    df_results = pd.DataFrame(results_summary).T
    print(df_results.to_string())
    
    # Save comparison
    df_results.to_csv('outputs/model_comparison.csv')
    
    print("\n✅ All training complete!")
    print(f"📁 Results saved in: outputs/")


if __name__ == '__main__':
    main()
