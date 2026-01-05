"""
Data Exploration for Pneumonia Detection Dataset
Run this to understand the dataset before training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from collections import Counter
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)


class DatasetExplorer:
    """Explore and visualize the chest X-ray dataset"""
    
    def __init__(self, data_dir='chest_xray'):
        self.data_dir = Path(data_dir)
        self.splits = ['train', 'val', 'test']
        self.classes = ['NORMAL', 'PNEUMONIA']
        
    def get_dataset_statistics(self):
        """Get basic statistics about the dataset"""
        print("=" * 60)
        print("📊 DATASET STATISTICS")
        print("=" * 60)
        
        stats = []
        
        for split in self.splits:
            for class_name in self.classes:
                class_path = self.data_dir / split / class_name
                
                if class_path.exists():
                    num_images = len(list(class_path.glob('*.jpeg')))
                    stats.append({
                        'Split': split.upper(),
                        'Class': class_name,
                        'Count': num_images
                    })
        
        df_stats = pd.DataFrame(stats)
        
        # Pivot for better view
        pivot = df_stats.pivot(index='Split', columns='Class', values='Count')
        pivot['Total'] = pivot.sum(axis=1)
        
        print("\n" + str(pivot))
        print("\n" + "=" * 60)
        
        # Calculate overall statistics
        print("\n📈 OVERALL STATISTICS:")
        print(f"   Total images: {pivot['Total'].sum():,}")
        print(f"   Total NORMAL: {pivot['NORMAL'].sum():,}")
        print(f"   Total PNEUMONIA: {pivot['PNEUMONIA'].sum():,}")
        print(f"   Class ratio (PNEUMONIA/NORMAL): {pivot['PNEUMONIA'].sum() / pivot['NORMAL'].sum():.2f}")
        
        return df_stats
    
    def visualize_class_distribution(self, stats_df):
        """Visualize class distribution across splits"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # By split
        pivot = stats_df.pivot(index='Split', columns='Class', values='Count')
        pivot.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
        axes[0].set_title('Images per Split and Class', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Split')
        axes[0].set_ylabel('Number of Images')
        axes[0].legend(title='Class')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Overall
        overall = stats_df.groupby('Class')['Count'].sum()
        colors = ['#3498db', '#e74c3c']
        axes[1].pie(overall, labels=overall.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
        axes[1].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/class_distribution.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved: outputs/class_distribution.png")
        plt.close()
    
    def visualize_sample_images(self, num_samples=5):
        """Display sample images from each class"""
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        
        for i, class_name in enumerate(self.classes):
            class_path = self.data_dir / 'train' / class_name
            images = list(class_path.glob('*.jpeg'))[:num_samples]
            
            for j, img_path in enumerate(images):
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')
                
                if j == 0:
                    axes[i, j].set_title(f'{class_name}\n',
                                        fontsize=12, fontweight='bold')
        
        plt.suptitle('Sample Chest X-Rays from Training Set',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('outputs/sample_images.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: outputs/sample_images.png")
        plt.close()
    
    def analyze_image_properties(self):
        """Analyze image dimensions and properties"""
        print("\n" + "=" * 60)
        print("🔍 IMAGE PROPERTIES ANALYSIS")
        print("=" * 60)
        
        widths, heights, aspects = [], [], []
        
        # Sample images from train set
        for class_name in self.classes:
            class_path = self.data_dir / 'train' / class_name
            images = list(class_path.glob('*.jpeg'))[:100]  # Sample 100 per class
            
            for img_path in images:
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                aspects.append(w / h)
        
        # Statistics
        print(f"\n📏 Dimension Statistics (sample of 200 images):")
        print(f"   Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
        print(f"   Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
        print(f"   Aspect ratio: mean={np.mean(aspects):.2f}")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(widths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title('Image Widths')
        axes[0].set_xlabel('Width (pixels)')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].hist(heights, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1].set_title('Image Heights')
        axes[1].set_xlabel('Height (pixels)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(axis='y', alpha=0.3)
        
        axes[2].hist(aspects, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[2].set_title('Aspect Ratios')
        axes[2].set_xlabel('Aspect Ratio (W/H)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Image Dimension Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/image_properties.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved: outputs/image_properties.png")
        plt.close()
    
    def visualize_intensity_distribution(self):
        """Analyze pixel intensity distributions"""
        print("\n" + "=" * 60)
        print("💡 PIXEL INTENSITY ANALYSIS")
        print("=" * 60)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for i, class_name in enumerate(self.classes):
            class_path = self.data_dir / 'train' / class_name
            images = list(class_path.glob('*.jpeg'))[:50]  # Sample 50 per class
            
            all_intensities = []
            
            for img_path in images:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                all_intensities.extend(img.flatten())
            
            axes[i].hist(all_intensities, bins=50, color='skyblue' if i == 0 else 'salmon',
                        edgecolor='black', alpha=0.7, density=True)
            axes[i].set_title(f'{class_name} - Pixel Intensity Distribution',
                             fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Pixel Intensity')
            axes[i].set_ylabel('Density')
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/intensity_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: outputs/intensity_distribution.png")
        plt.close()
    
    def check_data_quality(self):
        """Check for potential data quality issues"""
        print("\n" + "=" * 60)
        print("✓ DATA QUALITY CHECK")
        print("=" * 60)
        
        issues = []
        
        for split in self.splits:
            for class_name in self.classes:
                class_path = self.data_dir / split / class_name
                
                if not class_path.exists():
                    issues.append(f"Missing directory: {class_path}")
                    continue
                
                images = list(class_path.glob('*.jpeg'))
                
                for img_path in images[:100]:  # Check first 100
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            issues.append(f"Cannot read: {img_path}")
                        elif img.size == 0:
                            issues.append(f"Empty image: {img_path}")
                    except Exception as e:
                        issues.append(f"Error with {img_path}: {str(e)}")
        
        if issues:
            print(f"\n⚠️  Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"   - {issue}")
            if len(issues) > 10:
                print(f"   ... and {len(issues) - 10} more")
        else:
            print("\n✅ No data quality issues found!")
        
        return issues


def main():
    """Main exploration pipeline"""
    print("=" * 60)
    print("🔬 CHEST X-RAY DATASET EXPLORATION")
    print("=" * 60)
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Initialize explorer
    explorer = DatasetExplorer()
    
    # Get statistics
    stats = explorer.get_dataset_statistics()
    
    # Visualize distributions
    explorer.visualize_class_distribution(stats)
    
    # Show sample images
    explorer.visualize_sample_images(num_samples=5)
    
    # Analyze properties
    explorer.analyze_image_properties()
    
    # Intensity analysis
    explorer.visualize_intensity_distribution()
    
    # Quality check
    explorer.check_data_quality()
    
    print("\n" + "=" * 60)
    print("✅ EXPLORATION COMPLETE!")
    print("=" * 60)
    print("\n📁 All visualizations saved in: outputs/")
    print("\n💡 Key Takeaways:")
    print("   1. Check class imbalance - may need to handle during training")
    print("   2. Images have varying dimensions - preprocessing will standardize")
    print("   3. Review sample images to understand visual differences")
    print("   4. Consider data augmentation to improve model robustness")


if __name__ == '__main__':
    main()
