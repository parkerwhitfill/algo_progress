#!/usr/bin/env python3
"""
Batch score all rows with absolute algorithm quality
"""

import pandas as pd
import numpy as np
import time
import argparse
from score_absolute_innovations import setup_gemini, score_absolute_with_gemini

def main():
    parser = argparse.ArgumentParser(description='Batch absolute scoring')
    parser.add_argument('--api-key', required=True, help='Gemini API key')
    parser.add_argument('--delay', type=float, default=5.0, help='Delay between API calls')
    parser.add_argument('--checkpoint-every', type=int, default=25, help='Save checkpoint every N rows')
    
    args = parser.parse_args()
    
    # Setup
    model = setup_gemini(args.api_key)
    df = pd.read_csv('papers/dataset.csv')
    
    # Absolute score columns
    absolute_columns = [
        'optimizer_quality', 'architecture_quality', 'attention_mechanism',
        'training_efficiency', 'activation_functions', 'normalization_techniques',
        'learning_paradigm', 'regularization_techniques', 'special_algorithms',
        'overall_algorithm_quality'
    ]
    
    for col in absolute_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Track progress
    total_rows = len(df)
    processed = 0
    skipped = 0
    failed = 0
    
    print(f"Processing {total_rows} rows")
    print("="*60)
    
    for idx in range(total_rows):
        row = df.iloc[idx]
        algo_text = row['algorithmic']
        system_name = row['System']
        year = row.get('Year', 2020)
        
        # Skip if no algorithmic text
        if pd.isna(algo_text) or not algo_text.strip():
            continue
        
        # Skip if already scored
        if pd.notna(df.at[idx, 'optimizer_quality']):
            skipped += 1
            continue
        
        print(f"\n[{idx}/{total_rows}] {system_name} ({year})")
        
        try:
            # Get scores
            scores = score_absolute_with_gemini(model, algo_text, system_name, year)
            
            if scores:
                # Update dataframe
                for key, value in scores.items():
                    df.at[idx, key] = value
                processed += 1
                
                # Show key scores
                print(f"✓ Architecture: {scores['architecture_quality']}, "
                      f"Optimizer: {scores['optimizer_quality']}, "
                      f"Overall: {scores['overall_algorithm_quality']}")
            else:
                failed += 1
                print("✗ Failed to score")
                
        except Exception as e:
            failed += 1
            print(f"✗ Error: {str(e)[:100]}")
        
        # Save checkpoint
        if (processed + failed) % args.checkpoint_every == 0:
            df.to_csv('papers/dataset_with_absolute_scores.csv', index=False)
            print(f"\n💾 Saved checkpoint. Progress: {processed} done, {failed} failed, {skipped} skipped")
        
        # Delay
        time.sleep(args.delay)
    
    # Final save
    df.to_csv('papers/dataset_with_absolute_scores.csv', index=False)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Successfully scored: {processed}")
    print(f"Failed: {failed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Saved to: papers/dataset_with_absolute_scores.csv")
    
    # Show distribution
    if processed > 0:
        print("\nScore distributions:")
        for col in ['optimizer_quality', 'architecture_quality', 'overall_algorithm_quality']:
            mean = df[col].mean()
            print(f"{col}: mean={mean:.2f}")

if __name__ == "__main__":
    main()