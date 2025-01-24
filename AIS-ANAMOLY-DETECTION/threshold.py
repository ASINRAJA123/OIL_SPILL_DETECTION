import pandas as pd
import numpy as np

def extract_features(df):
    """
    Extract numerical features for classification from the dataframe.
    """
    feature_columns = [
        'Shape Factor 1', 'Shape Factor 2', 'Complexity',
        'Mean Intensity', 'Std Intensity', 'Power to Mean Ratio',
        'Mean Contrast', 'Max Contrast', 'Local Contrast',
        'Mean Gradient', 'Std Gradient', 'Max Gradient',
        'Mean Haralick Texture', 'Perimeter to Area'
    ]
    return df[feature_columns].dropna().to_numpy()

def calculate_confidence(row):
    """
    Calculate the confidence score for classification based on certain features.
    """
    # Thresholds based on known oil spill characteristics (these can be adjusted)
    intensity_threshold = 0.25
    complexity_threshold = 5.0
    
    # Confidence score calculation for oil spill
    oil_spill_confidence = 0
    if row['Mean Intensity'] > intensity_threshold:
        oil_spill_confidence += 0.5
    if row['Complexity'] > complexity_threshold:
        oil_spill_confidence += 0.5
        
    # Scale the confidence score between 0 and 1
    confidence = np.clip(oil_spill_confidence, 0, 1)
    
    return confidence

def classify_regions(df):
    """
    Classify regions and calculate confidence scores.
    """
    results = []
    for _, row in df.iterrows():
        # Calculate confidence score based on features
        confidence_score = calculate_confidence(row)
        
        # Classification: If the confidence score is greater than a threshold, classify as oil spill
        if confidence_score > 0.7:
            classification = 'Oil Spill'
            confidence = 'High'
        elif confidence_score > 0.4:
            classification = 'Non-Oil Spill'
            confidence = 'Medium'
        else:
            classification = 'Non-Oil Spill'
            confidence = 'Low'
        
        results.append({
            'Region': row['Region'],
            'Mask': row['Region'],  # Assuming 'Mask' corresponds to the Region for classification
            'Classification': classification,
            'Confidence': confidence,
            'Confidence_Score': confidence_score,  # Adding the confidence score
            'Area_km2': row['Area (km²)'],
            'Mean_Intensity': row['Mean Intensity'],
            'Complexity': row['Complexity']
        })
    return results

if __name__ == "__main__":

    # Example CSV file (Replace with the actual path)
    csv_file_path = r'F:\SIH_FINAL_AIS_SATE\comined.csv'

    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Clean and preprocess the data
    data = data.dropna(subset=['Mean Intensity', 'Complexity', '10U', '10V', 'SWH', 'Region'])

    # Classify regions
    results = classify_regions(data)

    # Display results with Confidence Score
    for result in results:
        print(result)
