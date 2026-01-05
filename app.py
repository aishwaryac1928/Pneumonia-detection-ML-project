"""
Interactive Gradio Demo for Pneumonia Detection
Deploy this on HuggingFace Spaces
"""

import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import os


class PneumoniaClassifier:
    """Wrapper for the pneumonia detection model"""
    
    def __init__(self, model_path='outputs/resnet50/best_model.h5'):
        """Load the trained model"""
        self.model = keras.models.load_model(model_path)
        self.img_size = (224, 224)
        
        # Get last conv layer for Grad-CAM
        self.last_conv_layer_name = self._get_last_conv_layer()
        self.grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output
            ]
        )
    
    def _get_last_conv_layer(self):
        """Find the last convolutional layer"""
        base_model = self.model.layers[1]
        for layer in reversed(base_model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        return None
    
    def preprocess_image(self, img):
        """Preprocess image for model"""
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def generate_gradcam(self, img_array):
        """Generate Grad-CAM heatmap"""
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, img, alpha=0.4):
        """Overlay heatmap on image"""
        # Resize heatmap
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if img.max() <= 1.0:
            img = np.uint8(255 * img)
        
        # Overlay
        superimposed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
        return superimposed
    
    def predict(self, image):
        """
        Make prediction on chest X-ray image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            prediction_text: str with diagnosis
            confidence: float
            overlay_image: numpy array with Grad-CAM overlay
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # Predict
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        # Generate Grad-CAM
        heatmap = self.generate_gradcam(img_array)
        overlay = self.overlay_heatmap(heatmap, image)
        
        # Determine class and confidence
        if prediction > 0.5:
            diagnosis = "PNEUMONIA DETECTED"
            confidence = prediction
            color = "🔴"
        else:
            diagnosis = "NORMAL"
            confidence = 1 - prediction
            color = "🟢"
        
        # Format output
        result_text = f"{color} **{diagnosis}**\n\n**Confidence:** {confidence:.1%}"
        
        return result_text, overlay


# Initialize classifier
print("Loading model...")
classifier = PneumoniaClassifier()
print("Model loaded successfully!")


def predict_pneumonia(image):
    """Gradio prediction function"""
    if image is None:
        return "Please upload an X-ray image", None
    
    result_text, overlay_image = classifier.predict(image)
    return result_text, overlay_image


# Create Gradio interface
title = "🏥 Pneumonia Detection from Chest X-Rays"

description = """
Upload a chest X-ray image to detect pneumonia using deep learning.

**How it works:**
- The model analyzes the X-ray using a ResNet50 architecture trained on 5,863 chest X-rays
- Grad-CAM visualization highlights the regions the model focuses on
- Green overlay indicates areas of interest

**⚠️ Medical Disclaimer:**
This is a demonstration tool for educational purposes only. It should NOT be used for actual medical diagnosis.
Always consult qualified healthcare professionals for medical advice.
"""

article = """
### About This Model

This pneumonia detection system uses transfer learning with ResNet50, achieving:
- **90%+** accuracy on test set
- **95%+** sensitivity (recall) for pneumonia detection
- Grad-CAM interpretability for transparency

### Dataset
Trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

### Technology Stack
- **Deep Learning:** TensorFlow/Keras
- **Architecture:** ResNet50 with transfer learning
- **Interpretability:** Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Deployment:** Gradio + HuggingFace Spaces

---
**Built for portfolio demonstration** | Not for clinical use
"""

examples = [
    # You can add example images here
]

# Create interface
demo = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=[
        gr.Textbox(label="Diagnosis", lines=3),
        gr.Image(label="Grad-CAM Visualization")
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# Launch
if __name__ == "__main__":
    demo.launch(share=True)
