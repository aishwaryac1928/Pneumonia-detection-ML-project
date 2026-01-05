"""
Master Script - Run Complete Pneumonia Detection Pipeline
This script runs the entire project from data exploration to demo
"""

import os
import sys
import time
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def run_step(step_name, script_name, skip=False):
    """Run a pipeline step"""
    print_header(f"STEP: {step_name}")
    
    if skip:
        print(f"⏭️  Skipping {step_name} (already completed)")
        return True
    
    print(f"🚀 Running: {script_name}")
    print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    # Run the script
    result = os.system(f"python {script_name}")
    
    elapsed = time.time() - start_time
    
    if result == 0:
        print(f"\n✅ Completed in {elapsed/60:.1f} minutes")
        return True
    else:
        print(f"\n❌ Failed! Check errors above.")
        return False


def check_requirements():
    """Check if all requirements are installed"""
    print_header("CHECKING REQUIREMENTS")
    
    required = ['tensorflow', 'keras', 'numpy', 'pandas', 'cv2', 
                'matplotlib', 'seaborn', 'gradio', 'PIL']
    
    missing = []
    
    package_display = {'tensorflow': 'tensorflow', 'keras': 'keras', 'numpy': 'numpy', 
                       'pandas': 'pandas', 'cv2': 'opencv-python', 'matplotlib': 'matplotlib',
                       'seaborn': 'seaborn', 'gradio': 'gradio', 'PIL': 'Pillow'}
    
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package_display[package]}")
        except ImportError:
            print(f"❌ {package_display[package]}")
            missing.append(package_display[package])
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All requirements installed!")
    return True


def check_dataset():
    """Check if dataset exists"""
    print_header("CHECKING DATASET")
    
    required_dirs = [
        'chest_xray/train/NORMAL',
        'chest_xray/train/PNEUMONIA',
        'chest_xray/test/NORMAL',
        'chest_xray/test/PNEUMONIA',
        'chest_xray/val/NORMAL',
        'chest_xray/val/PNEUMONIA'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing: {dir_path}")
            print("\n⚠️  Dataset not found!")
            print("Please download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
            return False
        else:
            num_files = len(list(Path(dir_path).glob('*.jpeg')))
            print(f"✅ {dir_path}: {num_files} images")
    
    print("\n✅ Dataset ready!")
    return True


def main():
    """Main pipeline"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║       PNEUMONIA DETECTION - COMPLETE PIPELINE                 ║
    ║                                                               ║
    ║   This script will run the entire project:                   ║
    ║   1. Data Exploration                                         ║
    ║   2. Model Training (ResNet50 + EfficientNetB0)              ║
    ║   3. Grad-CAM Visualization                                   ║
    ║   4. Model Comparison                                         ║
    ║   5. Launch Demo                                              ║
    ║                                                               ║
    ║   ⏰ Estimated time: 2-6 hours (depending on hardware)       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    QUICK_MODE = False  # Set to True for quick testing
    
    if QUICK_MODE:
        print("⚡ QUICK MODE ENABLED - Using reduced epochs for testing")
        print("   (Edit run_pipeline.py to disable)")
    
    # Ask user for confirmation
    response = input("\n🤔 Do you want to run the complete pipeline? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("👋 Exiting...")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check dataset
    if not check_dataset():
        return
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Pipeline steps
    steps = [
        ("Data Exploration", "explore_data.py", False),
        ("Model Training", "train.py", False),
        ("Grad-CAM Generation", "gradcam.py", False),
        ("Model Comparison", "utils.py", False)
    ]
    
    # Check what's already done
    skip_training = os.path.exists('outputs/resnet50/best_model.h5')
    skip_gradcam = os.path.exists('outputs/resnet50/gradcam')
    
    if skip_training:
        print("\n💡 Detected existing trained models. Options:")
        print("   1. Skip training (use existing models)")
        print("   2. Re-train from scratch")
        choice = input("Enter choice (1/2): ")
        if choice == '1':
            steps[1] = ("Model Training", "train.py", True)
            if skip_gradcam:
                steps[2] = ("Grad-CAM Generation", "gradcam.py", True)
    
    # Run pipeline
    print_header("STARTING PIPELINE")
    start_time = time.time()
    
    for step_name, script, skip in steps:
        success = run_step(step_name, script, skip)
        if not success:
            print(f"\n⚠️  Pipeline stopped at: {step_name}")
            print("Fix the errors and run again.")
            return
    
    total_time = time.time() - start_time
    
    # Summary
    print_header("PIPELINE COMPLETE! 🎉")
    print(f"✅ Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print("\n📁 Generated files:")
    print("   - outputs/resnet50/")
    print("   - outputs/efficientnetb0/")
    print("   - outputs/model_comparison.csv")
    print("   - outputs/model_comparison.png")
    
    print("\n🎯 Next Steps:")
    print("   1. Review outputs/ folder for results")
    print("   2. Check model_comparison.png for performance")
    print("   3. Run demo: python app.py")
    print("   4. Deploy to HuggingFace Spaces")
    
    # Ask if user wants to launch demo
    response = input("\n🚀 Launch demo now? (y/n): ")
    if response.lower() in ['y', 'yes']:
        print("\n🌐 Launching Gradio demo...")
        print("   Opening in browser at http://localhost:7860")
        os.system("python app.py")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("You can resume by running this script again")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Check the error message above for details")
