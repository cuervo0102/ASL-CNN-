import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

def extract_landmarks(image_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    try:
        image = cv2.imread(image_path)
        
        if image is None:
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            coords = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
            coords = np.array(coords).flatten()  # 63 features
            
            wrist = coords[:3]
            coords = coords - np.tile(wrist, 21)
            
            max_val = np.max(np.abs(coords))
            if max_val > 0:
                coords = coords / max_val
            
            hands.close()
            return coords
        
        hands.close()
        return None
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_dataset(dataset_path, output_path, max_images_per_class=400):
    """Process entire Kaggle dataset with optional limit per class"""
    
    os.makedirs(output_path, exist_ok=True)
    
    X, y = [], []
    label_map = {}
    current_label = 0
    
    if os.path.basename(dataset_path) == 'asl_alphabet_train':
        search_path = dataset_path
    elif os.path.exists(os.path.join(dataset_path, 'asl_alphabet_train')):
        search_path = os.path.join(dataset_path, 'asl_alphabet_train')
        print(f"Found dataset at: {search_path}")
    else:
        search_path = dataset_path
    
    all_items = os.listdir(search_path)
    folders = sorted([f for f in all_items 
                     if os.path.isdir(os.path.join(search_path, f)) 
                     and len(f) <= 3])  
    
    print(f"Found {len(folders)} letter folders: {folders}")
    
    for folder_name in folders:
        if folder_name.lower() in ['del', 'nothing', 'space']:
            print(f"Skipping folder: {folder_name}")
            continue
        
        folder_path = os.path.join(search_path, folder_name)
        label_map[current_label] = folder_name
        
        print(f"\nProcessing folder: {folder_name} (Label: {current_label})")
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) > max_images_per_class:
            print(f"  Found {len(image_files)} images, limiting to {max_images_per_class}")
            import random
            random.seed(42)  
            image_files = random.sample(image_files, max_images_per_class)
        
        success_count = 0
        fail_count = 0
        
        for img_name in tqdm(image_files, desc=f"Processing {folder_name}"):
            img_path = os.path.join(folder_path, img_name)
            landmarks = extract_landmarks(img_path)
            
            if landmarks is not None:
                X.append(landmarks)
                y.append(current_label)
                success_count += 1
            else:
                fail_count += 1
        
        print(f"  Success: {success_count}, Failed: {fail_count}")
        current_label += 1
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    if len(X) == 0:
        print("\nERROR: No samples extracted!")
        print("Please check:")
        print(f"1. Dataset path exists: {search_path}")
        print(f"2. Contains letter folders (A, B, C, etc.)")
        print(f"3. Each folder contains .jpg/.png images")
        return None, None, None
    
    np.save(f'{output_path}/X_landmarks.npy', X)
    np.save(f'{output_path}/y_labels.npy', y)
    
    import json
    with open(f'{output_path}/label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"Total samples saved: {len(X)}")
    print(f"Number of classes: {len(label_map)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Files saved to: {output_path}")
    print(f"{'='*50}")
    
    return X, y, label_map

if __name__ == "__main__":
    DATASET_PATH = r'C:\Users\DELL\Desktop\asl_alphabet_train\asl_alphabet_train'
    OUTPUT_PATH = 'processed_data'
    
    print("Starting landmark extraction...")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path does not exist: {DATASET_PATH}")
        exit(1)
    
    print(f"\nContents of {DATASET_PATH}:")
    try:
        items = os.listdir(DATASET_PATH)
        folders = [f for f in items if os.path.isdir(os.path.join(DATASET_PATH, f))]
        files = [f for f in items if os.path.isfile(os.path.join(DATASET_PATH, f))]
        print(f"  Folders ({len(folders)}): {folders[:10]}")
        if len(folders) > 10:
            print(f"  ... and {len(folders)-10} more")
        if files:
            print(f"  Files: {files[:5]}")
    except Exception as e:
        print(f"  Error listing contents: {e}")
    
    result = process_dataset(DATASET_PATH, OUTPUT_PATH, max_images_per_class=300)
    
    if result[0] is None:
        print("\nExtraction failed. Exiting.")
        exit(1)
    
    X, y, label_map = result
    
    print("\nLabel mapping:")
    for label, name in label_map.items():
        count = np.sum(y == label)
        print(f"  {label}: {name} ({count} samples)")