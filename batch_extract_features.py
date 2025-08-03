#!/usr/bin/env python3
"""
Batch extract algorithmic features from multiple papers
"""

import os
import sys
import time
import json
import pandas as pd
import google.generativeai as genai
from pathlib import Path
import argparse
from typing import Dict, Any, List
import traceback

from extract_algorithmic_features import (
    setup_gemini, 
    read_pdf_for_gemini, 
    create_prompt, 
    extract_features_from_paper,
    format_features_for_csv
)

def process_batch(model, df: pd.DataFrame, rows: List[int], papers_dir: str, 
                  output_file: str = None, delay: float = 2.0, skip_existing: bool = True):
    """Process multiple rows and save results"""
    results = []
    
    # Add algorithmic column if it doesn't exist
    if 'algorithmic' not in df.columns:
        df['algorithmic'] = ''
    
    # Track skipped rows
    skipped_count = 0
    
    for row_idx in rows:
        if row_idx >= len(df):
            print(f"Skipping row {row_idx} - out of bounds")
            continue
            
        row_data = df.iloc[row_idx].to_dict()
        system_name = row_data.get('System', f'Row {row_idx}')
        
        # Check if already processed
        if skip_existing and pd.notna(df.at[row_idx, 'algorithmic']) and df.at[row_idx, 'algorithmic'].strip():
            print(f"\nSkipping row {row_idx} ({system_name}) - already has algorithmic features")
            skipped_count += 1
            results.append({
                'row': row_idx,
                'system': system_name,
                'status': 'skipped_existing',
                'features': None
            })
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing row {row_idx}: {system_name}")
        print(f"{'='*60}")
        
        # Find PDF
        pdf_path = os.path.join(papers_dir, f"{row_idx}.pdf")
        if not os.path.exists(pdf_path):
            print(f"WARNING: PDF not found at {pdf_path}, skipping...")
            results.append({
                'row': row_idx,
                'system': system_name,
                'status': 'pdf_not_found',
                'features': None
            })
            continue
        
        try:
            # Upload PDF
            print(f"Uploading PDF: {pdf_path}")
            pdf_file = read_pdf_for_gemini(pdf_path)
            
            # Extract features
            print("Extracting features...")
            features = extract_features_from_paper(model, pdf_file, row_data)
            
            if "error" in features:
                print(f"Error: {features['error']}")
                results.append({
                    'row': row_idx,
                    'system': system_name,
                    'status': 'extraction_error',
                    'features': features
                })
            else:
                # Format for CSV
                csv_string = format_features_for_csv(features)
                df.at[row_idx, 'algorithmic'] = csv_string
                
                # Save immediately after each successful extraction
                if output_file:
                    df.to_csv(output_file, index=False)
                    print(f"\nSaved progress to {output_file}")
                
                print("\nExtracted features:")
                print(json.dumps(features, indent=2))
                
                results.append({
                    'row': row_idx,
                    'system': system_name,
                    'status': 'success',
                    'features': features,
                    'csv_string': csv_string
                })
            
            # Clean up
            pdf_file.delete()
            
        except Exception as e:
            print(f"ERROR processing row {row_idx}: {str(e)}")
            traceback.print_exc()
            results.append({
                'row': row_idx,
                'system': system_name,
                'status': 'exception',
                'error': str(e)
            })
        
        # Delay between API calls
        if row_idx != rows[-1]:  # Don't delay after last row
            print(f"\nWaiting {delay} seconds before next request...")
            time.sleep(delay)
    
    # Save results
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\n\nSaved updated dataset to: {output_file}")
    
    # Summary
    print("\n\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"Total rows processed: {len(results)}")
    print(f"Successful extractions: {success_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Failed extractions: {len(results) - success_count - skipped_count}")
    
    print("\nDetailed results:")
    for result in results:
        if result['status'] == 'success':
            status_emoji = "✅"
        elif result['status'] == 'skipped_existing':
            status_emoji = "⏭️"
        else:
            status_emoji = "❌"
        print(f"{status_emoji} Row {result['row']} ({result['system']}): {result['status']}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Batch extract algorithmic features')
    parser.add_argument('--api-key', required=True, help='Gemini API key')
    parser.add_argument('--start-row', type=int, default=0, help='Starting row (inclusive)')
    parser.add_argument('--end-row', type=int, default=None, help='Ending row (exclusive)')
    parser.add_argument('--rows', type=str, help='Comma-separated list of specific rows')
    parser.add_argument('--csv-path', default='papers/dataset.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--papers-dir', default='papers',
                        help='Directory containing PDFs')
    parser.add_argument('--output', help='Output CSV path (default: overwrite input)')
    parser.add_argument('--delay', type=float, default=2.0, 
                        help='Delay between API calls in seconds')
    parser.add_argument('--force', action='store_true',
                        help='Force re-processing of rows that already have algorithmic features')
    
    args = parser.parse_args()
    
    # Setup Gemini
    print("Setting up Gemini API...")
    model = setup_gemini(args.api_key)
    
    # Read dataset
    print(f"Reading dataset from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"Dataset has {len(df)} rows")
    
    # Determine which rows to process
    if args.rows:
        rows = [int(r.strip()) for r in args.rows.split(',')]
        print(f"Processing specific rows: {rows}")
    else:
        start = args.start_row
        end = args.end_row if args.end_row is not None else len(df)
        rows = list(range(start, end))
        print(f"Processing rows {start} to {end-1}")
    
    # Process batch
    output_file = args.output or args.csv_path
    skip_existing = not args.force
    results = process_batch(model, df, rows, args.papers_dir, output_file, args.delay, skip_existing)
    
    print("\nDone!")

if __name__ == "__main__":
    main()