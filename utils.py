"""
Utility functions for pneumonia detection project
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras


def load_model_safely(model_path):
    """Load a Keras model with error handling"""
    try:
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def compare_models(results_dir='outputs'):
    """Compare performance of different models"""
    results_dir = Path(results_dir)
    
    models_data = []
    
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir():
            results_file = model_dir / 'results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                models_data.append({
                    'Model': model_dir.name.upper(),
                    'Accuracy': results['accuracy'],
                    'Precision': results['PNEUMONIA']['precision'],
                    'Recall': results['PNEUMONIA']['recall'],
                    'F1-Score': results['PNEUMONIA']['f1-score']
                })
    
    if models_data:
        import pandas as pd
        df = pd.DataFrame(models_data)
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
        
        return df
    else:
        print("⚠️  No results found. Train models first!")
        return None


def plot_comparison(results_dir='outputs'):
    """Create comparison plots for different models"""
    df = compare_models(results_dir)
    
    if df is None:
        return
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(df['Model'], df[metric], color=['#3498db', '#e74c3c'][:len(df)])
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(results_dir) / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {results_dir}/model_comparison.png")
    plt.close()


def get_class_weights(train_dir):
    """Calculate class weights for handling imbalance"""
    from pathlib import Path
    
    normal_count = len(list(Path(train_dir).glob('NORMAL/*.jpeg')))
    pneumonia_count = len(list(Path(train_dir).glob('PNEUMONIA/*.jpeg')))
    
    total = normal_count + pneumonia_count
    
    # Class weights inversely proportional to frequencies
    weight_normal = total / (2 * normal_count)
    weight_pneumonia = total / (2 * pneumonia_count)
    
    class_weights = {
        0: weight_normal,  # NORMAL
        1: weight_pneumonia  # PNEUMONIA
    }
    
    print("\n📊 Class Distribution:")
    print(f"   NORMAL: {normal_count} ({normal_count/total*100:.1f}%)")
    print(f"   PNEUMONIA: {pneumonia_count} ({pneumonia_count/total*100:.1f}%)")
    print(f"\n⚖️  Class Weights:")
    print(f"   NORMAL: {weight_normal:.3f}")
    print(f"   PNEUMONIA: {weight_pneumonia:.3f}")
    
    return class_weights


def create_submission_report(model_name='resnet50', output_dir='outputs'):
    """Generate a comprehensive report for the model"""
    output_dir = Path(output_dir) / model_name
    
    if not output_dir.exists():
        print(f"❌ Model directory not found: {output_dir}")
        return
    
    report = []
    report.append("="*60)
    report.append(f"PNEUMONIA DETECTION - {model_name.upper()} MODEL REPORT")
    report.append("="*60)
    report.append("")
    
    # Load results
    results_file = output_dir / 'results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        report.append("📊 Performance Metrics:")
        report.append(f"   Overall Accuracy: {results['accuracy']:.4f}")
        report.append("")
        report.append("   Class: NORMAL")
        report.append(f"      Precision: {results['NORMAL']['precision']:.4f}")
        report.append(f"      Recall:    {results['NORMAL']['recall']:.4f}")
        report.append(f"      F1-Score:  {results['NORMAL']['f1-score']:.4f}")
        report.append("")
        report.append("   Class: PNEUMONIA")
        report.append(f"      Precision: {results['PNEUMONIA']['precision']:.4f}")
        report.append(f"      Recall:    {results['PNEUMONIA']['recall']:.4f}")
        report.append(f"      F1-Score:  {results['PNEUMONIA']['f1-score']:.4f}")
    
    report.append("")
    report.append("="*60)
    report.append("Files Generated:")
    report.append(f"   ✓ Model: best_model.h5")
    report.append(f"   ✓ Results: results.json")
    report.append(f"   ✓ Confusion Matrix: confusion_matrix.png")
    report.append(f"   ✓ ROC Curve: roc_curve.png")
    report.append(f"   ✓ Training History: training_history.png")
    report.append(f"   ✓ Grad-CAM Visualizations: gradcam/")
    report.append("="*60)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report
    with open(output_dir / 'report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\n✅ Report saved: {output_dir}/report.txt")


def predict_single_image(model_path, image_path, img_size=(224, 224)):
    """Make prediction on a single image"""
    # Load model
    model = load_model_safely(model_path)
    if model is None:
        return None
    
    # Load and preprocess image
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    if prediction > 0.5:
        diagnosis = "PNEUMONIA"
        confidence = prediction
    else:
        diagnosis = "NORMAL"
        confidence = 1 - prediction
    
    result = {
        'diagnosis': diagnosis,
        'confidence': confidence,
        'raw_prediction': prediction
    }
    
    print(f"\n🔍 Prediction for: {image_path}")
    print(f"   Diagnosis: {diagnosis}")
    print(f"   Confidence: {confidence:.2%}")
    
    return result


if __name__ == '__main__':
    # Example usage
    print("🛠️  Utility Functions Demo")
    print("="*60)
    
    # Compare models if results exist
    if os.path.exists('outputs'):
        compare_models()
        plot_comparison()
    
    # Calculate class weights
    if os.path.exists('chest_xray/train'):
        get_class_weights('chest_xray/train')
