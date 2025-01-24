import zipfile
import os
import re

def unzip_safe_file(safe_file_path, extract_to_dir):
    extract_to_dir = 'extracted_SAR_data_check_no_spill' 
    with zipfile.ZipFile(safe_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)
    print(f"Extracted to: {extract_to_dir}")
    
def extract_date(path):
    # Search for the date in the string
    date_pattern = r"(\d{4})(\d{2})(\d{2})"
    match = re.search(date_pattern, path)

    if match:
        # Extract year, month, and day
        year, month, day = match.groups()
        return day,month,year
    else:
        return None,None,None
def modify_path(path):
    # Replace 'measurement' with 'annotation'
    updated_path = path.replace("measurement", "annotation")
    # Find the position of the word 'annotation' and truncate everything after it
    annotation_index = updated_path.find("annotation")
    if annotation_index != -1:
        # Keep everything up to 'annotation'
        updated_path = updated_path[:annotation_index + len("annotation")]
    return updated_path
    