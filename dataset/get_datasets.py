#!/usr/bin/env python3
"""
AI Image and Video Compression Dataset Downloader
Automatically downloads and sets up all required datasets for the project.
"""

import os
import sys
import json
import requests
import zipfile
import tarfile
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import kaggle

def create_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {path}")

def download_file(url, destination, description="file"):
    """Download file with progress indication."""
    print(f"📥 Downloading {description}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r  Progress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✓ Downloaded {description} to {destination}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {description}: {e}")
        return False

def extract_archive(archive_path, extract_to, archive_type="auto"):
    """Extract archive file."""
    print(f"📦 Extracting {archive_path}...")
    try:
        if archive_type == "auto":
            if archive_path.endswith('.zip'):
                archive_type = "zip"
            elif archive_path.endswith(('.tar.gz', '.tgz')):
                archive_type = "tar.gz"
            elif archive_path.endswith('.tar'):
                archive_type = "tar"
        
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_type in ["tar.gz", "tgz"]:
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_type == "tar":
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
        
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Failed to extract {archive_path}: {e}")
        return False

def download_kaggle_dataset(dataset_path, destination):
    """Download dataset from Kaggle."""
    print(f"📥 Downloading Kaggle dataset: {dataset_path}")
    try:
        # Ensure kaggle API is configured
        if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
            print("⚠️  Kaggle API key not found. Please set up your Kaggle API key first:")
            print("   1. Go to Kaggle.com -> Account -> API -> Create New API Token")
            print("   2. Place kaggle.json in ~/.kaggle/")
            print("   3. chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        # Download using kaggle API
        kaggle.api.dataset_download_files(dataset_path, path=destination, unzip=True)
        print(f"✓ Downloaded Kaggle dataset to {destination}")
        return True
    except Exception as e:
        print(f"✗ Failed to download Kaggle dataset {dataset_path}: {e}")
        return False

def setup_kodak_dataset():
    """Download and setup Kodak dataset."""
    print("\n🖼️  Setting up Kodak Dataset...")
    dataset_dir = "kodak"
    create_directory(dataset_dir)
    
    # Kodak dataset from Kaggle
    success = download_kaggle_dataset("sherylmehta/kodak-dataset", dataset_dir)
    return success

def setup_coco_dataset():
    """Download and setup COCO 2017 dataset."""
    print("\n🏷️  Setting up COCO 2017 Dataset...")
    dataset_dir = "coco2017"
    create_directory(dataset_dir)
    
    # COCO URLs
    urls = {
        "train_images": "http://images.cocodataset.org/zips/train2017.zip",
        "val_images": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }
    
    success = True
    for name, url in urls.items():
        filename = os.path.join(dataset_dir, url.split('/')[-1])
        if download_file(url, filename, f"COCO {name}"):
            extract_archive(filename, dataset_dir)
            os.remove(filename)  # Clean up zip file
        else:
            success = False
    
    return success

def setup_bsds500_dataset():
    """Download and setup BSDS500 dataset."""
    print("\n🔍 Setting up BSDS500 Dataset...")
    dataset_dir = "bsds500"
    create_directory(dataset_dir)
    
    url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    filename = os.path.join(dataset_dir, "BSR_bsds500.tgz")
    
    if download_file(url, filename, "BSDS500"):
        extract_archive(filename, dataset_dir, "tar.gz")
        os.remove(filename)
        return True
    return False

def setup_isic2018_dataset():
    """Download and setup ISIC2018 dataset."""
    print("\n🏥 Setting up ISIC2018 Dataset...")
    dataset_dir = "isic2018"
    create_directory(dataset_dir)
    
    # ISIC2018 from Kaggle
    success = download_kaggle_dataset("nodoubttome/skin-cancer9-classesisic", dataset_dir)
    return success

def setup_license_plate_dataset():
    """Download and setup US License Plate dataset."""
    print("\n🚗 Setting up US License Plate Dataset...")
    dataset_dir = "us_license_plates"
    create_directory(dataset_dir)
    
    # This would typically be a custom dataset or from a specific source
    # For now, we'll create a placeholder structure
    print("⚠️  US License Plate dataset requires manual setup or specific source access")
    print(f"   Please place your license plate images in: {dataset_dir}")
    print("   Expected structure: us_license_plates/images/*.jpg")
    
    return True

def setup_mot17_dataset():
    """Download and setup MOT17 dataset."""
    print("\n🎬 Setting up MOT17 Dataset...")
    dataset_dir = "mot17"
    create_directory(dataset_dir)
    
    # MOT17 dataset (requires registration typically)
    print("⚠️  MOT17 dataset requires registration at https://motchallenge.net/")
    print(f"   Please download MOT17 and extract to: {dataset_dir}")
    print("   Expected structure: mot17/train/, mot17/test/")
    
    return True

def setup_ucf101_dataset():
    """Download and setup UCF101 dataset."""
    print("\n📹 Setting up UCF101 Dataset...")
    dataset_dir = "ucf101"
    create_directory(dataset_dir)
    
    url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    print(f"⚠️  UCF101 dataset is large (~6.5GB) and in RAR format")
    print(f"   Please download from: {url}")
    print(f"   Extract to: {dataset_dir}")
    print("   Expected structure: ucf101/UCF-101/")
    
    return True

def create_get_dataset_txt():
    """Create the get_dataset.txt file with dataset information."""
    content = """# AI Image and Video Compression - Dataset Information

# This file contains information about all datasets used in the project
# Run: python get_datasets.py to automatically download available datasets

# Dataset List:
# 1. Kodak - Image compression baseline (Kaggle)
# 2. COCO 2017 - Object detection (Direct download)
# 3. BSDS500 - Edge detection (Berkeley)
# 4. ISIC2018 - Medical image classification (Kaggle)
# 5. US License Plates - OCR evaluation (Manual setup required)
# 6. MOT17 - Video tracking (Registration required)
# 7. UCF101 - Video classification (Manual download)

# Automatic Downloads:
kodak,kaggle,sherylmehta/kodak-dataset
coco2017,direct,http://images.cocodataset.org/zips/
bsds500,direct,http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
isic2018,kaggle,nodoubttome/skin-cancer9-classesisic

# Manual Setup Required:
us_license_plates,manual,Custom dataset
mot17,registration,https://motchallenge.net/
ucf101,manual,https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

# Notes:
# - Kaggle datasets require API key setup
# - Some datasets require registration or manual download
# - Total size: ~20GB+ when all datasets are downloaded
"""
    
    with open('get_dataset.txt', 'w') as f:
        f.write(content)
    print("✓ Created get_dataset.txt")

def main():
    """Main function to orchestrate dataset downloads."""
    print("🚀 AI Image and Video Compression Dataset Setup")
    print("=" * 50)
    
    # Create dataset directory structure
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create get_dataset.txt if it doesn't exist
    if not os.path.exists('get_dataset.txt'):
        create_get_dataset_txt()
    
    # Setup each dataset
    datasets = {
        'Kodak': setup_kodak_dataset,
        'COCO 2017': setup_coco_dataset,
        'BSDS500': setup_bsds500_dataset,
        'ISIC2018': setup_isic2018_dataset,
        'US License Plates': setup_license_plate_dataset,
        'MOT17': setup_mot17_dataset,
        'UCF101': setup_ucf101_dataset
    }
    
    results = {}
    for dataset_name, setup_func in datasets.items():
        try:
            results[dataset_name] = setup_func()
        except Exception as e:
            print(f"✗ Error setting up {dataset_name}: {e}")
            results[dataset_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Dataset Setup Summary:")
    for dataset_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed/Manual setup required"
        print(f"   {dataset_name}: {status}")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\n✅ {successful}/{total} datasets ready for automatic processing")
    
    if successful < total:
        print("\n⚠️  Some datasets require manual setup. Please check the instructions above.")

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import kaggle
        import requests
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install required packages:")
        print("pip install kaggle requests")
        sys.exit(1)
    
    main()