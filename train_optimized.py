"""
OPTIMIZED Pneumonia Detection Training - Reduced Disk Usage
This version implements memory and disk usage optimizations
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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

# Set memory growth for GPU (if available)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Limit TensorFlow memory usage
tf.config.set_soft_device_placement(True)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


class OptimizedPneumoniaDetector:
    """Memory-optimized pneumonia detector"""
    
    def __init__(self, data_dir='chest_xray', img_size=(224, 224), batch_size=16):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size  # Reduced from 32 to 16
        self.model = None
        self.history = None
        self.model_name = None
        
        # Create outputs directory
        self.output_dir = Path('outputs')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔧 Optimized settings:")
        print(f"   Batch size: {self.batch_size} (reduced for memory)")
        print(f"   Image size: {self.img_size}")
        
    def prepare_data(self):
        """Prepare data generators with reduced memory usage"""
        print("🔄 Preparing optimized data generators...")
        
        # Training data augmentation (lighter)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,  # Reduced from 15
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation and test data
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators with prefetch
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
        
    def build_model(self, architecture='resnet50', learning_rate=0.0001):
        """Build model with specified architecture"""
        self.model_name = architecture
        print(f"\n🏗️  Building {architecture.upper()} model...")
        
        # Clear any existing models from memory
        keras.backend.clear_session()
        gc.collect()
        
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
        
        # Freeze base model
        base_model.trainable = False
        
        # Build model with smaller dense layers
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)  # Reduced from 256
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)   # Reduced from 128
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile with mixed precision if supported
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
    
    def train(self, epochs=10, fine_tune_epochs=5):
        """Train with reduced epochs and better checkpointing"""
        print(f"\n🚀 Starting optimized training for {self.model_name}...")
        print(f"⚠️  Using reduced epochs: {epochs} + {fine_tune_epochs}")
        
        # Create model directory
        model_dir = self.output_dir / self.model_name
        model_dir.mkdir(exist_ok=True)
        
        # Optimized callbacks - less frequent saving
        callbacks = [
            ModelCheckpoint(
                str(model_dir / 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_freq='epoch',  # Save only at epoch end
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=3,  # Reduced from 5
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,  # Reduced from 3
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(
                str(model_dir / 'training_log.csv'),
                append=False
            )
        ]
        
        # Phase 1: Train with frozen base
        print("\n📊 Phase 1: Training with frozen base...")
        history1 = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            workers=2,  # Limit workers
            use_multiprocessing=False  # Safer on Windows
        )
        
        # Clear memory
        gc.collect()
        keras.backend.clear_session()
        
        # Phase 2: Fine-tuning
        print("\n📊 Phase 2: Fine-tuning...")
        
        # Unfreeze top layers only
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze most layers
        for layer in base_model.layers[:int(len(base_model.layers) * 0.85)]:
            layer.trainable = False
        
        # Recompile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-6),
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
            verbose=1,
            workers=2,
            use_multiprocessing=False
        )
        
        # Combine histories
        self.history = {
            key: history1.history[key] + history2.history[key]
            for key in history1.history.keys()
        }
        
        # Save final model
        self.model.save(str(model_dir / 'final_model.h5'))
        
        print(f"\n✅ Training completed!")
        
        # Clear memory
        gc.collect()
        
        return self.history
    
    def evaluate(self):
        """Evaluate model"""
        print(f"\n📈 Evaluating {self.model_name}...")
        
        model_dir = self.output_dir / self.model_name
        
        # Get predictions in batches to save memory
        self.test_generator.reset()
        predictions = []
        
        for i in range(len(self.test_generator)):
            batch_pred = self.model.predict(
                self.test_generator[i][0],
                verbose=0
            )
            predictions.extend(batch_pred)
            
            # Clear memory periodically
            if i % 10 == 0:
                gc.collect()
        
        predictions = np.array(predictions)
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
        plt.savefig(model_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
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
        plt.savefig(model_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Evaluation complete! AUC: {roc_auc:.4f}")
        
        # Clear memory
        gc.collect()
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        model_dir = self.output_dir / self.model_name
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
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
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle(f'{self.model_name.upper()} - Training History')
        plt.tight_layout()
        plt.savefig(model_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training history saved!")


def main():
    """Main training pipeline - ONE MODEL ONLY"""
    print("=" * 60)
    print("🏥 OPTIMIZED PNEUMONIA DETECTION TRAINING")
    print("=" * 60)
    
    # Choose architecture
    print("\n📋 SELECT MODEL TO TRAIN:")
    print("   1. ResNet50 (recommended)")
    print("   2. EfficientNetB0")
    
    choice = input("\n   Enter choice (1/2): ")
    
    if choice == '1':
        architecture = 'resnet50'
    elif choice == '2':
        architecture = 'efficientnetb0'
    else:
        print("Invalid choice!")
        return
    
    print(f"\n🔧 Training {architecture.upper()} with optimizations:")
    print("   • Reduced batch size (16)")
    print("   • Reduced epochs (10+5)")
    print("   • Less frequent disk writes")
    print("   • Memory cleanup enabled")
    print("   • Smaller dense layers")
    
    response = input("\n   Continue? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("👋 Exiting...")
        return
    
    # Initialize detector
    detector = OptimizedPneumoniaDetector(batch_size=16)
    
    # Prepare data
    detector.prepare_data()
    
    # Build model
    detector.build_model(architecture=architecture)
    
    # Train
    detector.train(epochs=10, fine_tune_epochs=5)
    
    # Plot history
    detector.plot_training_history()
    
    # Evaluate
    results = detector.evaluate()
    
    print("\n\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Results saved in: outputs/{architecture}/")
    print("\n💡 Next steps:")
    print("   1. Check outputs folder for results")
    print("   2. Run app.py to test the model")
    print("   3. Train the other model if needed")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
        print("Model checkpoint saved in outputs/")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        gc.collect()
        keras.backend.clear_session()
