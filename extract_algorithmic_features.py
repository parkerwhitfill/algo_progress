#!/usr/bin/env python3
"""
Extract algorithmic features from papers using Gemini API
"""

import os
import sys
import json
import pandas as pd
import google.generativeai as genai
from pathlib import Path
import argparse
from typing import Dict, Any

# Configure Gemini API
def setup_gemini(api_key: str):
    """Initialize Gemini with API key"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash-exp')

def read_pdf_for_gemini(pdf_path: str):
    """Upload PDF to Gemini for processing"""
    # Upload the file
    uploaded_file = genai.upload_file(pdf_path)
    print(f"Uploaded file: {uploaded_file.name}")
    return uploaded_file

def create_prompt(row_data: Dict[str, Any]) -> str:
    """Create the prompt for Gemini based on dataset row"""
    prompt = f"""
You are an expert in machine learning and natural language processing. I need you to extract algorithmic features from a research paper about language models.

Context from dataset:
- System Name: {row_data.get('System', 'Unknown')}
- Authors: {row_data.get('Author(s)', 'Unknown')}
- Year: {row_data.get('Year', 'Unknown')}
- Title: {row_data.get('Reference', 'Unknown')}
- Architecture Type: {row_data.get('Architecture', 'Unknown')}
- Base Model: {row_data.get('Base Model', 'Unknown') if pd.notna(row_data.get('Base Model')) else 'None'}

Please read this paper and extract ONLY the algorithmic features used in training the language model described. Focus on:

1. **Architecture Details**: Specific architecture type and key components (e.g., "2-layer LSTM", "SRU++ with 10 layers", "Transformer with 12 layers")
2. **Optimizer**: The optimization algorithm used (e.g., "SGD with momentum", "Adam", "RAdam")
3. **Learning Rate Schedule**: How the learning rate changes during training (e.g., "cosine annealing", "linear decay", "constant")
4. **Training Techniques**: Special training methods (e.g., "gradient clipping", "dropout", "layer normalization")
5. **Attention Mechanisms**: If applicable, describe attention details (e.g., "single-head attention", "multi-head with 8 heads")
6. **Regularization**: Methods to prevent overfitting (e.g., "weight decay 0.1", "dropout 0.2")
7. **Special Algorithms**: Any novel algorithms introduced (e.g., "CD-GraB for example ordering", "PairBalance algorithm")
8. **Initialization**: How weights are initialized (if mentioned)

Do NOT include:
- Hardware details (GPUs, TPUs)
- Dataset information (size, name)
- Performance metrics (perplexity, accuracy)
- Parameter counts
- Training time or epochs
- Non-algorithmic implementation details

Provide the output as a JSON object with the following structure:
```json
{{
    "architecture": "description of architecture",
    "optimizer": "optimizer details",
    "learning_rate_schedule": "LR schedule",
    "training_techniques": ["technique1", "technique2"],
    "attention": "attention details if applicable",
    "regularization": ["regularization1", "regularization2"],
    "special_algorithms": ["algorithm1", "algorithm2"],
    "initialization": "initialization method if mentioned",
    "other_algorithmic_features": ["any other relevant algorithmic features"]
}}
```

Return ONLY the JSON object, no additional text.
"""
    return prompt

def extract_features_from_paper(model, pdf_file, row_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract algorithmic features using Gemini"""
    prompt = create_prompt(row_data)
    
    # Generate response
    response = model.generate_content([prompt, pdf_file])
    
    try:
        # Extract JSON from response
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        features = json.loads(response_text.strip())
        return features
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response was: {response.text}")
        return {"error": "Failed to parse response", "raw_response": response.text}

def format_features_for_csv(features: Dict[str, Any]) -> str:
    """Format extracted features as a single string for CSV"""
    parts = []
    
    if features.get("architecture"):
        parts.append(f"Architecture: {features['architecture']}")
    
    if features.get("optimizer"):
        parts.append(f"Optimizer: {features['optimizer']}")
    
    if features.get("learning_rate_schedule"):
        parts.append(f"LR Schedule: {features['learning_rate_schedule']}")
    
    if features.get("training_techniques"):
        techniques = ", ".join(features['training_techniques'])
        parts.append(f"Training: {techniques}")
    
    if features.get("attention"):
        parts.append(f"Attention: {features['attention']}")
    
    if features.get("regularization"):
        reg = ", ".join(features['regularization'])
        parts.append(f"Regularization: {reg}")
    
    if features.get("special_algorithms"):
        algos = ", ".join(features['special_algorithms'])
        parts.append(f"Special Algorithms: {algos}")
    
    if features.get("initialization"):
        parts.append(f"Initialization: {features['initialization']}")
    
    if features.get("other_algorithmic_features"):
        others = ", ".join(features['other_algorithmic_features'])
        parts.append(f"Other: {others}")
    
    return "; ".join(parts)

def main():
    parser = argparse.ArgumentParser(description='Extract algorithmic features from papers')
    parser.add_argument('--api-key', required=True, help='Gemini API key')
    parser.add_argument('--row', type=int, default=1, help='Row number to process (0-indexed)')
    parser.add_argument('--csv-path', default='papers/dataset.csv', 
                        help='Path to dataset CSV')
    parser.add_argument('--papers-dir', default='papers',
                        help='Directory containing PDFs')
    parser.add_argument('--test', action='store_true', help='Test mode - just print results')
    
    args = parser.parse_args()
    
    # Setup Gemini
    model = setup_gemini(args.api_key)
    
    # Read dataset
    df = pd.read_csv(args.csv_path)
    
    # Get row data
    if args.row >= len(df):
        print(f"Error: Row {args.row} not found in dataset (max: {len(df)-1})")
        return
    
    row_data = df.iloc[args.row].to_dict()
    print(f"\nProcessing row {args.row}: {row_data.get('System', 'Unknown')}")
    
    # Find corresponding PDF
    pdf_path = os.path.join(args.papers_dir, f"{args.row}.pdf")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return
    
    print(f"Reading PDF: {pdf_path}")
    
    # Upload PDF to Gemini
    pdf_file = read_pdf_for_gemini(pdf_path)
    
    # Extract features
    print("\nExtracting algorithmic features...")
    features = extract_features_from_paper(model, pdf_file, row_data)
    
    if "error" in features:
        print(f"Error extracting features: {features['error']}")
        return
    
    # Format for display
    print("\n=== Extracted Algorithmic Features ===")
    print(json.dumps(features, indent=2))
    
    # Format for CSV
    csv_string = format_features_for_csv(features)
    print(f"\n=== Formatted for CSV ===")
    print(csv_string)
    
    if not args.test:
        # Update the CSV with the new column
        if 'algorithmic' not in df.columns:
            df['algorithmic'] = ''
        
        df.at[args.row, 'algorithmic'] = csv_string
        df.to_csv(args.csv_path, index=False)
        print(f"\nUpdated row {args.row} in {args.csv_path}")
    else:
        print("\n(Test mode - CSV not updated)")
    
    # Clean up uploaded file
    pdf_file.delete()

if __name__ == "__main__":
    main()