"""
Prediction Script - Test the trained model on new images
"""

import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow import keras
import cv2


def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """Load and preprocess a single image"""
    # Load image
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    
    # Convert to array
    img_array = keras.preprocessing.image.img_to_array(img)
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img


def predict_image(model_path, image_path):
    """Make prediction on a single image"""
    print(f"\n{'='*60}")
    print("🔍 PNEUMONIA PREDICTION")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # Load and preprocess image
    print(f"\n📷 Loading image: {image_path}")
    img_array, original_img = load_and_preprocess_image(image_path)
    
    # Make prediction
    print("🤖 Making prediction...")
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Interpret results
    if prediction > 0.5:
        diagnosis = "PNEUMONIA DETECTED"
        confidence = prediction
        color = 'red'
        emoji = '🔴'
    else:
        diagnosis = "NORMAL"
        confidence = 1 - prediction
        color = 'green'
        emoji = '🟢'
    
    # Display results
    print(f"\n{'='*60}")
    print(f"{emoji} DIAGNOSIS: {diagnosis}")
    print(f"{'='*60}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw prediction: {prediction:.4f}")
    print(f"{'='*60}\n")
    
    # Visualize
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Chest X-Ray', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Prediction bar
    plt.subplot(1, 2, 2)
    categories = ['NORMAL', 'PNEUMONIA']
    probabilities = [1-prediction, prediction]
    colors_bar = ['green', 'red']
    
    bars = plt.barh(categories, probabilities, color=colors_bar, alpha=0.7)
    plt.xlim([0, 1])
    plt.xlabel('Probability', fontweight='bold')
    plt.title(f'Prediction: {diagnosis}\nConfidence: {confidence:.1%}',
             fontsize=12, fontweight='bold', color=color)
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        plt.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = Path('outputs') / 'prediction_result.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Visualization saved: {output_path}")
    
    plt.show()
    
    return {
        'diagnosis': diagnosis,
        'confidence': confidence,
        'prediction': prediction
    }


def batch_predict(model_path, image_dir):
    """Make predictions on multiple images"""
    print(f"\n{'='*60}")
    print("🔍 BATCH PREDICTION")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # Get all images
    image_dir = Path(image_dir)
    image_paths = list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    print(f"\n📷 Found {len(image_paths)} images")
    
    results = []
    
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {img_path.name}")
        
        # Preprocess
        img_array, _ = load_and_preprocess_image(img_path)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Interpret
        if prediction > 0.5:
            diagnosis = "PNEUMONIA"
            confidence = prediction
        else:
            diagnosis = "NORMAL"
            confidence = 1 - prediction
        
        results.append({
            'image': img_path.name,
            'diagnosis': diagnosis,
            'confidence': confidence,
            'raw_prediction': prediction
        })
        
        print(f"   → {diagnosis} ({confidence:.1%})")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 BATCH RESULTS SUMMARY")
    print(f"{'='*60}")
    
    normal_count = sum(1 for r in results if r['diagnosis'] == 'NORMAL')
    pneumonia_count = len(results) - normal_count
    
    print(f"Total images: {len(results)}")
    print(f"NORMAL: {normal_count} ({normal_count/len(results)*100:.1f}%)")
    print(f"PNEUMONIA: {pneumonia_count} ({pneumonia_count/len(results)*100:.1f}%)")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    output_path = Path('outputs') / 'batch_predictions.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Predict pneumonia from chest X-rays')
    parser.add_argument('--model', type=str, default='outputs/resnet50/best_model.h5',
                       help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Path to directory of images')
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("Train the model first using: python train.py")
        return
    
    # Single image or batch
    if args.image:
        predict_image(args.model, args.image)
    elif args.batch:
        batch_predict(args.model, args.batch)
    else:
        print("Usage:")
        print("  Single image: python predict.py --image path/to/image.jpg")
        print("  Batch:        python predict.py --batch path/to/directory/")
        print("\nExample:")
        print("  python predict.py --image chest_xray/test/PNEUMONIA/person1_virus_6.jpeg")


if __name__ == '__main__':
    main()
