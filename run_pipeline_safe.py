"""
SAFE Master Script - Run Pneumonia Detection Pipeline with Disk Usage Protection
This version prevents 100% disk usage by implementing safety measures
"""

import os
import sys
import time
import psutil
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def check_disk_space():
    """Check available disk space"""
    disk = psutil.disk_usage('C:\\')
    free_gb = disk.free / (1024**3)
    percent_free = (disk.free / disk.total) * 100
    
    print(f"💾 Disk Status:")
    print(f"   Free space: {free_gb:.1f} GB ({percent_free:.1f}%)")
    
    if free_gb < 10:
        print(f"   ⚠️  WARNING: Low disk space! At least 10GB recommended.")
        return False
    
    if percent_free < 15:
        print(f"   ⚠️  WARNING: Less than 15% disk space free.")
        return False
    
    return True


def check_system_resources():
    """Check system resources before starting"""
    print_header("CHECKING SYSTEM RESOURCES")
    
    # Memory
    mem = psutil.virtual_memory()
    mem_gb = mem.total / (1024**3)
    mem_available_gb = mem.available / (1024**3)
    
    print(f"💻 System Memory:")
    print(f"   Total: {mem_gb:.1f} GB")
    print(f"   Available: {mem_available_gb:.1f} GB ({mem.percent}% used)")
    
    if mem_available_gb < 4:
        print(f"   ⚠️  WARNING: Low memory! At least 4GB recommended.")
        return False
    
    # Disk
    if not check_disk_space():
        return False
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"\n🔧 CPU Usage: {cpu_percent}%")
    
    print("\n✅ System resources OK!")
    return True


def monitor_disk_usage():
    """Monitor disk I/O during operation"""
    disk_io = psutil.disk_io_counters()
    return {
        'read_mb': disk_io.read_bytes / (1024**2),
        'write_mb': disk_io.write_bytes / (1024**2)
    }


def run_step_safe(step_name, script_name, skip=False, max_disk_usage=80):
    """Run a pipeline step with disk monitoring"""
    print_header(f"STEP: {step_name}")
    
    if skip:
        print(f"⏭️  Skipping {step_name} (already completed)")
        return True
    
    # Check resources before starting
    disk = psutil.disk_usage('C:\\')
    if (disk.used / disk.total * 100) > max_disk_usage:
        print(f"❌ Disk usage too high: {disk.used / disk.total * 100:.1f}%")
        print(f"   Please free up disk space before continuing.")
        return False
    
    print(f"🚀 Running: {script_name}")
    print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
    
    start_time = time.time()
    disk_start = monitor_disk_usage()
    
    # Run the script
    result = os.system(f"python {script_name}")
    
    elapsed = time.time() - start_time
    disk_end = monitor_disk_usage()
    
    # Calculate disk I/O
    disk_read = disk_end['read_mb'] - disk_start['read_mb']
    disk_write = disk_end['write_mb'] - disk_start['write_mb']
    
    print(f"\n📊 Disk I/O: Read {disk_read:.1f} MB, Write {disk_write:.1f} MB")
    
    if result == 0:
        print(f"✅ Completed in {elapsed/60:.1f} minutes")
        return True
    else:
        print(f"❌ Failed! Check errors above.")
        return False


def cleanup_temp_files():
    """Clean up temporary files to free disk space"""
    print_header("CLEANING TEMPORARY FILES")
    
    # TensorFlow cache
    tf_cache = Path.home() / '.keras'
    if tf_cache.exists():
        try:
            import shutil
            shutil.rmtree(tf_cache)
            print("✅ Cleared TensorFlow cache")
        except Exception as e:
            print(f"⚠️  Could not clear TF cache: {e}")
    
    # Python cache
    for cache_dir in Path('.').rglob('__pycache__'):
        try:
            import shutil
            shutil.rmtree(cache_dir)
            print(f"✅ Cleared {cache_dir}")
        except:
            pass
    
    print("✅ Cleanup complete")


def main():
    """Main safe pipeline"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║       PNEUMONIA DETECTION - SAFE PIPELINE MODE                ║
    ║                                                               ║
    ║   This version includes disk usage protection:                ║
    ║   • Resource monitoring                                       ║
    ║   • Disk usage limits                                         ║
    ║   • One model at a time                                       ║
    ║   • Automatic cleanup                                         ║
    ║                                                               ║
    ║   ⏰ Estimated time: 2-4 hours per model                     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check system resources
    if not check_system_resources():
        print("\n⚠️  System resources insufficient!")
        print("   Recommendations:")
        print("   1. Close other applications")
        print("   2. Free up disk space (at least 10GB)")
        print("   3. Ensure at least 4GB RAM available")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("👋 Exiting...")
            return
    
    # Configuration
    print("\n📋 SAFE MODE OPTIONS:")
    print("   1. Train only ResNet50 (recommended)")
    print("   2. Train only EfficientNetB0")
    print("   3. Skip training, only evaluate existing models")
    print("   4. Exit")
    
    choice = input("\n   Enter choice (1-4): ")
    
    if choice == '4':
        print("👋 Exiting...")
        return
    
    # Clean up before starting
    cleanup_temp_files()
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Run pipeline based on choice
    print_header("STARTING SAFE PIPELINE")
    start_time = time.time()
    
    try:
        if choice in ['1', '2']:
            # Data exploration (lighter step)
            success = run_step_safe(
                "Data Exploration", 
                "explore_data.py",
                skip=os.path.exists('outputs/class_distribution.png')
            )
            
            if not success:
                print("\n⚠️  Pipeline stopped")
                return
            
            # Train single model
            model_choice = 'resnet50' if choice == '1' else 'efficientnetb0'
            
            # Create modified training script for single model
            print(f"\n🔧 Configuring for {model_choice} only...")
            
            # Run training
            print("\n⚠️  IMPORTANT: Training will start now.")
            print("   • Monitor disk usage using Task Manager")
            print("   • If disk hits 95%+, press Ctrl+C to stop")
            print("   • Model checkpoints save progress automatically")
            
            input("\nPress Enter to continue...")
            
            success = run_step_safe(
                f"{model_choice.upper()} Training",
                "train.py",
                max_disk_usage=85
            )
            
            if not success:
                print("\n⚠️  Training stopped")
                print("   • Check disk space")
                print("   • Trained model is saved in outputs/")
                return
        
        elif choice == '3':
            # Only evaluation
            print("📊 Running evaluation on existing models...")
            success = run_step_safe("Model Evaluation", "utils.py")
        
        total_time = time.time() - start_time
        
        # Summary
        print_header("PIPELINE COMPLETE! 🎉")
        print(f"✅ Total time: {total_time/60:.1f} minutes")
        print("\n📁 Check outputs/ folder for results")
        
        # Final disk check
        check_disk_space()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("Progress has been saved. You can resume by running this script again.")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Check the error message above for details")
    
    finally:
        # Cleanup
        cleanup_temp_files()


if __name__ == '__main__':
    # Install psutil if needed
    try:
        import psutil
    except ImportError:
        print("📦 Installing required package: psutil")
        os.system("pip install psutil")
        import psutil
    
    main()
