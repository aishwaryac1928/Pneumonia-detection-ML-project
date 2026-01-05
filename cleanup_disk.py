"""
Disk Space Cleanup Helper
Identifies large files and directories to help free up space
"""

import os
from pathlib import Path
import shutil


def get_size(path):
    """Get size of file or directory in GB"""
    try:
        if os.path.isfile(path):
            return os.path.getsize(path) / (1024**3)
        else:
            total = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total += os.path.getsize(filepath)
                    except:
                        pass
            return total / (1024**3)
    except:
        return 0


def find_large_items(start_path, min_size_gb=1.0):
    """Find files and folders larger than min_size_gb"""
    large_items = []
    
    try:
        for item in Path(start_path).iterdir():
            try:
                size_gb = get_size(item)
                if size_gb >= min_size_gb:
                    large_items.append((str(item), size_gb))
            except:
                pass
    except:
        pass
    
    return sorted(large_items, key=lambda x: x[1], reverse=True)


def main():
    print("=" * 70)
    print("🔍 DISK SPACE ANALYSIS")
    print("=" * 70)
    
    # Check common locations
    locations = [
        (Path.home() / "Downloads", "Downloads"),
        (Path.home() / "Videos", "Videos"),
        (Path.home() / "Documents", "Documents"),
        (Path.home() / "Pictures", "Pictures"),
        (Path.home() / "Desktop", "Desktop"),
        (Path("C:/Users/aishwaryac/AppData/Local/Temp"), "Temp Files"),
    ]
    
    print("\n📁 Large Items by Location:\n")
    
    total_found = 0
    
    for location, name in locations:
        if location.exists():
            print(f"\n{name} ({location}):")
            large_items = find_large_items(location, min_size_gb=0.5)
            
            if large_items:
                for path, size in large_items[:5]:  # Top 5
                    print(f"  • {size:.2f} GB - {Path(path).name}")
                    total_found += size
            else:
                print("  No large items (>500MB)")
    
    print("\n" + "=" * 70)
    print(f"📊 Total large items found: {total_found:.2f} GB")
    print("=" * 70)
    
    # Recommendations
    print("\n💡 CLEANUP RECOMMENDATIONS:\n")
    
    recommendations = [
        "1. Empty Recycle Bin (right-click → Empty Recycle Bin)",
        "2. Delete old downloads you don't need",
        "3. Move videos/photos to external drive or cloud",
        "4. Clear browser cache (Chrome: Ctrl+Shift+Del)",
        "5. Uninstall unused programs (Settings → Apps)",
        "6. Run Disk Cleanup (search 'Disk Cleanup' in Windows)",
        "7. Delete old Windows updates (Disk Cleanup → Clean up system files)",
    ]
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n🎯 GOAL: Free up at least 30GB (target 50GB+ for comfort)")
    
    # Check specific temp locations
    print("\n" + "=" * 70)
    print("🗑️  SAFE TO DELETE LOCATIONS:")
    print("=" * 70)
    
    temp_locations = [
        (Path.home() / "AppData/Local/Temp", "Windows Temp"),
        (Path("C:/Windows/Temp"), "System Temp"),
        (Path.home() / ".keras", "Keras Cache"),
        (Path.home() / ".cache", "Python Cache"),
    ]
    
    for location, name in temp_locations:
        if location.exists():
            size = get_size(location)
            if size > 0.1:
                print(f"\n{name}:")
                print(f"  Location: {location}")
                print(f"  Size: {size:.2f} GB")
                print(f"  Safe to delete: YES")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
