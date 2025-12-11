import os
import random
import shutil
import glob
from tqdm import tqdm
import sys

# Add project root to python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src import config
from src.core.roi import ROIExtractor
import cv2

class DataProcessor:
    def __init__(self, source_dirs, target_dir, train_ratio=0.8, roi_extract=True):
        """
        Args:
            source_dirs (list): List of source directories (e.g. ['dataset/session1', 'dataset/session2'])
            target_dir (str): Target directory (e.g. 'dataset_roi')
            train_ratio (float): Ratio of training data
            roi_extract (bool): Whether to extract ROI during copy
        """
        self.source_dirs = source_dirs
        self.target_dir = target_dir
        self.train_dir = os.path.join(target_dir, 'train')
        self.valid_dir = os.path.join(target_dir, 'valid')
        self.train_ratio = train_ratio
        self.roi_extract = roi_extract
        self.roi_extractor = ROIExtractor() if roi_extract else None

    def _get_all_images(self):
        """
        Collect all images from source directories and group by person ID.
        Assumes flat structure: session/00001.tiff
        Every 10 images correspond to one person.
        """
        person_images = {}
        
        for source_dir in self.source_dirs:
            if not os.path.exists(source_dir):
                print(f"Warning: Source directory not found: {source_dir}")
                continue
                
            print(f"Scanning {source_dir}...")
            
            # Get all images
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']:
                images.extend(glob.glob(os.path.join(source_dir, ext)))
            
            for img_path in images:
                try:
                    filename = os.path.basename(img_path)
                    name_no_ext = os.path.splitext(filename)[0]
                    
                    # Extract number from filename (e.g. '00001' -> 1)
                    img_idx = int(name_no_ext)
                    
                    # Calculate person ID: every 10 images is one person
                    # 1-10 -> person 0, 11-20 -> person 1
                    person_id = (img_idx - 1) // 10
                    person_str = f"{person_id:04d}" # e.g. 0000
                    
                    if person_str not in person_images:
                        person_images[person_str] = []
                    
                    person_images[person_str].append(img_path)
                    
                except ValueError:
                    print(f"Warning: Could not parse index from filename {os.path.basename(img_path)}, skipping.")
                    continue
        
        return person_images

    def _process_single_image(self, src_path, dst_path):
        if self.roi_extract:
            try:
                img = cv2.imread(src_path)
                if img is None:
                    return False
                
                # Extract ROI
                # Note: Assuming extract_roi returns (center, roi_img, rect) based on src/core/roi.py
                # We only need the roi_img
                _, roi_img, _ = self.roi_extractor.extract_roi(img, visualize=False)
                
                if roi_img is not None:
                    cv2.imwrite(dst_path, roi_img)
                    return True
                else:
                    # Fallback: copy original if ROI extraction fails? 
                    # Or skip? Let's skip to keep dataset clean, or maybe copy original.
                    # Ideally we want only valid ROIs.
                    # For now, let's skip and log warning
                    print(f"ROI extraction failed for {src_path}")
                    return False
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
                return False
        else:
            shutil.copy(src_path, dst_path)
            return True

    def run(self):
        # 1. Collect images
        person_images = self._get_all_images()
        print(f"Found {len(person_images)} persons.")
        
        if not person_images:
            print("No images found!")
            return

        # 2. Prepare target directories
        if os.path.exists(self.target_dir):
            print(f"Cleaning target directory: {self.target_dir}")
            shutil.rmtree(self.target_dir)
        
        os.makedirs(self.train_dir)
        os.makedirs(self.valid_dir)

        # 3. Split and Process
        total_processed = 0
        total_failed = 0
        
        print("Processing images...")
        for person, images in tqdm(person_images.items()):
            # Create person dir in train and valid
            os.makedirs(os.path.join(self.train_dir, person), exist_ok=True)
            os.makedirs(os.path.join(self.valid_dir, person), exist_ok=True)
            
            # Shuffle images for random split
            random.shuffle(images)
            
            # Split
            split_idx = int(len(images) * self.train_ratio)
            train_imgs = images[:split_idx]
            valid_imgs = images[split_idx:]
            
            # Process Train
            for img_path in train_imgs:
                fname = os.path.basename(img_path)
                # Add prefix to avoid name collision if sessions have same filenames
                # Current structure: raw_dataset/session1/00001.tiff
                # os.path.dirname(img_path) -> raw_dataset/session1
                # os.path.basename(...) -> session1
                session_name = os.path.basename(os.path.dirname(img_path))
                dst_name = f"{session_name}_{fname}"
                
                # Handle different extensions if needed, but ROI extractor saves as is or we force jpg/png?
                # cv2.imwrite uses extension to decide format. 
                # If input is tiff, we might want to save as jpg or png for size/compat, or keep tiff.
                # Let's keep original extension for now.
                
                dst_path = os.path.join(self.train_dir, person, dst_name)
                
                if self._process_single_image(img_path, dst_path):
                    total_processed += 1
                else:
                    total_failed += 1

            # Process Valid
            for img_path in valid_imgs:
                fname = os.path.basename(img_path)
                session_name = os.path.basename(os.path.dirname(img_path))
                dst_name = f"{session_name}_{fname}"
                dst_path = os.path.join(self.valid_dir, person, dst_name)
                
                if self._process_single_image(img_path, dst_path):
                    total_processed += 1
                else:
                    total_failed += 1

        print(f"\nDone!")
        print(f"Total processed: {total_processed}")
        print(f"Total failed: {total_failed}")
        print(f"Train set: {self.train_dir}")
        print(f"Valid set: {self.valid_dir}")

if __name__ == "__main__":
    # Configuration
    SOURCE_DIRS = [
        os.path.join("raw_dataset", "session1"),
        os.path.join("raw_dataset", "session2")
    ]
    TARGET_DIR = "dataset_raw_split"
    
    processor = DataProcessor(
        source_dirs=SOURCE_DIRS,
        target_dir=TARGET_DIR,
        train_ratio=0.8,
        roi_extract=False  # Set to True to extract ROI, False to just copy
    )
    processor.run()
