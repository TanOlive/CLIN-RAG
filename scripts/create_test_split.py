import os
import shutil
import pandas as pd
from pathlib import Path

def main():
    # 1. Setup Paths
    root_dir = Path(__file__).resolve().parent.parent
    reports_path = root_dir / 'data' / 'archive' / 'indiana_reports.csv'
    images_dir = root_dir / 'data' / 'archive' / 'images' / 'images_normalized'
    target_dir = root_dir / 'data' / 'test_samples'
    
    # Create or clear target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in target_dir.iterdir():
        if f.is_file():
            try:
                f.unlink()
            except PermissionError:
                pass
    
    # 2. Stratified Sampling Logic
    print("Reading reports...")
    df = pd.read_csv(reports_path)
    
    # Pathological keywords to exclude from Normal
    patho_keywords = [
        'opacity', 'consolidation', 'infiltrate', 'atelectasis',
        'cardiomegaly', 'enlarged heart', 'cardiac silhouette enlarged',
        'effusion', 'pleural',
        'fracture', 'emphysema', 'nodule', 'scarring'
    ]
    
    def get_category(row):
        text = f"{str(row['findings'])} {str(row['impression'])}".lower()
        
        # Check categories in order
        if any(k in text for k in ['normal', 'clear', 'unremarkable']) and not any(k in text for k in patho_keywords):
            return "Normal"
        elif any(k in text for k in ['opacity', 'consolidation', 'infiltrate', 'atelectasis']):
            return "Opacity_Consolidation"
        elif any(k in text for k in ['cardiomegaly', 'enlarged heart', 'cardiac silhouette enlarged']):
            return "Cardiomegaly"
        elif any(k in text for k in ['effusion', 'pleural']):
            return "Pleural_Effusion"
        elif any(k in text for k in ['fracture', 'emphysema', 'nodule', 'scarring']):
            return "Other_Pathology"
        return None

    df['Category'] = df.apply(get_category, axis=1)
    
    # 3. Selection
    categories = ["Normal", "Opacity_Consolidation", "Cardiomegaly", "Pleural_Effusion", "Other_Pathology"]
    selected_uids = []
    
    print("Sampling categories...")
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        if len(cat_df) >= 5:
            sampled = cat_df.sample(n=5, random_state=42)
        else:
            sampled = cat_df
        selected_uids.append(sampled)
        
    final_df = pd.concat(selected_uids)
    print(f"Total selected cases: {len(final_df)}")
    
    # 4. File Operations & 5. Metadata Output
    metadata_records = []
    category_counts = {cat: 0 for cat in categories}
    
    print("Copying images...")
    for _, row in final_df.iterrows():
        uid = row['uid']
        category = row['Category']
        
        # Find all images for this UID
        # Based on file listing, images start with "UID_"
        prefix = f"{uid}_"
        matched_images = list(images_dir.glob(f"{prefix}*.png"))
        
        if not matched_images:
            # Maybe the images use a slightly different naming convention or don't exist
            print(f"Warning: No images found for UID {uid}")
            continue
            
        for img_path in matched_images:
            original_filename = img_path.name
            new_filename = f"{category}_{original_filename}"
            target_path = target_dir / new_filename
            
            shutil.copy2(img_path, target_path)
            category_counts[category] += 1
            
            # Save metadata per image copied (or you could save per UID, but per image is often better for CV tasks)
            metadata_records.append({
                'uid': uid,
                'Category': category,
                'original_filename': original_filename,
                'new_filename': new_filename,
                'findings': row['findings'],
                'impression': row['impression']
            })

    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(target_dir / 'test_metadata.csv', index=False)
    print(f"Test metadata saved to {target_dir / 'test_metadata.csv'} with {len(metadata_df)} image records.")
    
    # 6. Console Output
    print("\n--- Summary of Copied Images ---")
    for cat, count in category_counts.items():
        print(f"{cat}: {count} images")

if __name__ == "__main__":
    main()
