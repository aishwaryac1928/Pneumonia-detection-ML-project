"""
Grad-CAM Visualization for Pneumonia Detection
Generates heatmaps showing what the model focuses on
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os


class GradCAM:
    """Grad-CAM implementation for CNN visualization"""
    
    def __init__(self, model, last_conv_layer_name):
        """
        Args:
            model: Trained Keras model
            last_conv_layer_name: Name of the last convolutional layer
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        
        # Create a model that maps the input image to the activations
        # of the last conv layer and the output predictions
        self.grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output
            ]
        )
    
    def generate_heatmap(self, img_array, pred_index=None):
        """Generate Grad-CAM heatmap for an image"""
        
        # Compute gradient of top predicted class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Gradient of the output neuron with respect to the output feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Mean intensity of the gradient over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel by importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, img, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Overlay heatmap on original image"""
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Convert to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is in uint8 format
        if img.max() <= 1.0:
            img = np.uint8(255 * img)
        
        # Superimpose heatmap on image
        superimposed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
        
        return superimposed


def preprocess_image(img_path, img_size=(224, 224)):
    """Load and preprocess image"""
    img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img


def visualize_prediction(model, img_path, gradcam, output_path=None, img_size=(224, 224)):
    """Complete visualization pipeline for a single image"""
    
    # Preprocess
    img_array, original_img = preprocess_image(img_path, img_size)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    class_name = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    # Generate heatmap
    heatmap = gradcam.generate_heatmap(img_array)
    
    # Overlay
    original_img_array = np.array(original_img)
    superimposed = gradcam.overlay_heatmap(heatmap, original_img_array)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original X-Ray', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(superimposed)
    axes[2].set_title(f'Prediction: {class_name}\nConfidence: {confidence:.2%}',
                     fontsize=12, fontweight='bold',
                     color='red' if class_name == 'PNEUMONIA' else 'green')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    plt.close()
    
    return {
        'prediction': class_name,
        'confidence': confidence,
        'heatmap': heatmap,
        'overlay': superimposed
    }


def batch_visualize(model, test_dir, gradcam, num_samples=10, output_dir='outputs/gradcam'):
    """Generate Grad-CAM visualizations for multiple test images"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎨 Generating Grad-CAM visualizations...")
    
    # Get sample images from each class
    normal_dir = Path(test_dir) / 'NORMAL'
    pneumonia_dir = Path(test_dir) / 'PNEUMONIA'
    
    normal_images = list(normal_dir.glob('*.jpeg'))[:num_samples//2]
    pneumonia_images = list(pneumonia_dir.glob('*.jpeg'))[:num_samples//2]
    
    all_images = normal_images + pneumonia_images
    
    results = []
    for i, img_path in enumerate(all_images):
        true_label = 'NORMAL' if 'NORMAL' in str(img_path) else 'PNEUMONIA'
        output_path = output_dir / f'gradcam_{true_label}_{i:02d}.png'
        
        result = visualize_prediction(model, str(img_path), gradcam, str(output_path))
        result['true_label'] = true_label
        result['image_path'] = str(img_path)
        results.append(result)
        
        print(f"  [{i+1}/{len(all_images)}] True: {true_label} | "
              f"Predicted: {result['prediction']} ({result['confidence']:.1%})")
    
    print(f"\n✅ Generated {len(results)} Grad-CAM visualizations!")
    print(f"📁 Saved in: {output_dir}")
    
    return results


def get_last_conv_layer_name(model, architecture='resnet50'):
    """Get the name of the last convolutional layer for different architectures"""
    
    if architecture == 'resnet50':
        # For ResNet50, the last conv layer is in the base model
        base_model = model.layers[1]
        for layer in reversed(base_model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
    
    elif architecture == 'efficientnetb0':
        # For EfficientNetB0
        base_model = model.layers[1]
        for layer in reversed(base_model.layers):
            if 'conv' in layer.name.lower() or 'block' in layer.name.lower():
                return layer.name
    
    # Fallback: find last Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    
    raise ValueError("Could not find a convolutional layer in the model")


def main():
    """Main Grad-CAM visualization pipeline"""
    
    print("=" * 60)
    print("🔍 GRAD-CAM VISUALIZATION PIPELINE")
    print("=" * 60)
    
    # Configuration
    architectures = ['resnet50', 'efficientnetb0']
    test_dir = 'chest_xray/test'
    
    for arch in architectures:
        print(f"\n\n{'='*60}")
        print(f"Processing {arch.upper()}")
        print(f"{'='*60}\n")
        
        model_path = f'outputs/{arch}/best_model.h5'
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found: {model_path}")
            print(f"   Please train the model first using train.py")
            continue
        
        # Load model
        print(f"📦 Loading model from: {model_path}")
        model = keras.models.load_model(model_path)
        
        # Get last conv layer name
        last_conv_layer = get_last_conv_layer_name(model, arch)
        print(f"🎯 Using last conv layer: {last_conv_layer}")
        
        # Create GradCAM
        gradcam = GradCAM(model, last_conv_layer)
        
        # Generate visualizations
        output_dir = f'outputs/{arch}/gradcam'
        batch_visualize(model, test_dir, gradcam, num_samples=20, output_dir=output_dir)
    
    print("\n✅ All Grad-CAM visualizations complete!")


if __name__ == '__main__':
    main()
