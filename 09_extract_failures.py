"""
Phase 9: Edge Case Analysis 
Team: Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak
Description: Uses pandas to sort the batch results and physically isolates the X-Ray images that triggered the worst (lowest) VCE scores for manual radiologist/human review.
"""
import pandas as pd
import os
import shutil

def extract_lowest_vce_cases(num_cases=10):
    print("Loading VCE Audit Data...")
    df = pd.read_csv("vce_audit_results.csv")
    
    # Sort the dataframe by the lowest VCE scores (targeting VCE Inversions / Text-Prior Hijacks)
    df_sorted = df.sort_values(by="VCE_Score", ascending=True)
    lowest_cases = df_sorted.head(num_cases)
    
    # Create an output directory
    output_dir = "Lowest_VCE_Cases"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Extracting Top {num_cases} Lowest VCE Images ---")
    
    # In the Kaggle dataset, the images are split into NORMAL and PNEUMONIA folders
    base_dir = "chest_xray/test"
    
    for index, row in lowest_cases.iterrows():
        filename = row['Filename']
        ground_truth = row['Ground_Truth']
        vce_score = row['VCE_Score']
        
        # Construct the original path
        original_path = os.path.join(base_dir, ground_truth, filename)
        
        # Construct the new destination path (embedding the VCE score in the filename for easy tracking)
        new_filename = f"VCE_{vce_score:.4f}_{ground_truth}_{filename}"
        destination_path = os.path.join(output_dir, new_filename)
        
        try:
            shutil.copy2(original_path, destination_path)
            print(f"Copied: {new_filename}")
        except FileNotFoundError:
            print(f"Could not find: {original_path}")

    print(f"\nDone! Open the '{output_dir}' folder to view the images.")

if __name__ == "__main__":
    extract_lowest_vce_cases()