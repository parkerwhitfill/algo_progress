#!/usr/bin/env python
"""
Download arXiv papers with simple row number naming
"""

import pandas as pd
import os
import re
import time
import requests
import argparse

def load_dataset():
    """Load the algorithmic progress dataset from Google Sheets"""
    print("Loading dataset from Google Sheets...")
    path = "https://docs.google.com/spreadsheets/d/11m8O_mU0cUkOB_5wluPne4PNsuvsKNbbVAzbYNy-NXY/edit#gid=2087221150"
    path = path.replace("edit#", "export?") + "&format=csv"
    df = pd.read_csv(path, parse_dates=True)
    print(f"Loaded {len(df)} entries")
    return df

def extract_arxiv_id(url):
    """Extract arXiv ID from various URL formats"""
    if pd.isna(url):
        return None
    
    url = str(url)
    
    # Pattern for arXiv URLs
    patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',  # New format: 1234.56789
        r'arxiv\.org/abs/([a-zA-Z\-]+/\d{7})',  # Old format: cs.CL/0701234
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5})',  # PDF links
        r'arxiv\.org/pdf/([a-zA-Z\-]+/\d{7})',  # Old PDF format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def download_paper(arxiv_id, output_path):
    """Download a paper from arXiv given its ID"""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return True, "Success"
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Download arXiv papers with simple naming')
    parser.add_argument('--output-dir', default='papers', help='Output directory for PDFs')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between downloads (seconds)')
    parser.add_argument('--limit', type=int, help='Limit number of papers to download')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    df = load_dataset()
    
    # Save the dataset locally for reference
    dataset_path = os.path.join(args.output_dir, 'dataset.csv')
    df.to_csv(dataset_path, index=True)  # Include index for easy reference
    print(f"\nSaved dataset to: {dataset_path}")
    
    # Extract arXiv IDs
    print(f"\nExtracting arXiv IDs...")
    df['arxiv_id'] = df['Link'].apply(extract_arxiv_id)
    
    # Get indices of rows with valid arXiv IDs
    valid_arxiv = df[df['arxiv_id'].notna()]
    
    if args.limit:
        valid_arxiv = valid_arxiv.head(args.limit)
    
    print(f"Found {len(valid_arxiv)} papers to download")
    
    # Download papers
    print(f"\nDownloading papers to {args.output_dir}/...")
    successful = 0
    failed = 0
    
    for i, (idx, row) in enumerate(valid_arxiv.iterrows()):
        arxiv_id = row['arxiv_id']
        
        # Simple filename: just the row number
        filename = f"{idx}.pdf"
        filepath = os.path.join(args.output_dir, filename)
        
        print(f"\n[{i+1}/{len(valid_arxiv)}] Row {idx}: {arxiv_id} ({row['System']})")
        
        success, msg = download_paper(arxiv_id, filepath)
        
        if success:
            print(f"  ✓ Saved as: {filename}")
            successful += 1
        else:
            print(f"  ✗ Failed: {msg}")
            failed += 1
        
        # Delay between downloads
        if i < len(valid_arxiv) - 1:
            time.sleep(args.delay)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {args.output_dir}/")
    print(f"\nUsage: PDF for row N is at '{args.output_dir}/N.pdf'")

if __name__ == "__main__":
    main()