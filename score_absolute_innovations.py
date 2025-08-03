#!/usr/bin/env python3
"""
Score algorithmic features against 2025 frontier (absolute scale)
10 = Best known algorithm in 2025
"""

import pandas as pd
import numpy as np
import google.generativeai as genai
import json
import time
import argparse

def setup_gemini(api_key: str):
    """Initialize Gemini with API key"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash-exp')

def create_absolute_scoring_prompt(algo_text, system_name, year):
    """Create prompt for absolute scoring against 2025 frontier"""
    prompt = f"""
You are an expert in AI/ML algorithms. Score the following system on an ABSOLUTE scale where 10 represents the best known algorithm in 2025.

System: {system_name} (Year: {year})
Algorithmic features: {algo_text}

Score each dimension on an absolute scale (1-10) where 10 = state-of-the-art in 2025:

1. **Optimizer Quality** (1-10)
   - 1-2: Vanilla SGD
   - 3-4: SGD with momentum
   - 5: RMSprop
   - 6-7: Adam/AdamW
   - 8: RAdam, LAMB
   - 9: Lion, Sophia
   - 10: Best optimizer in 2025

2. **Architecture Quality** (1-10)
   - 1-2: Simple RNN
   - 3-4: LSTM/GRU
   - 5: Basic CNN
   - 6-7: Transformer
   - 8: Efficient Transformers (Flash, Linformer)
   - 9: Mamba, RWKV
   - 10: Best architecture in 2025

3. **Attention Mechanism** (0-10)
   - 0: No attention
   - 3: Basic attention
   - 5: Multi-head attention
   - 7: Efficient attention (Flash, sparse)
   - 8: Linear complexity attention
   - 9: State-space models replacing attention
   - 10: Best attention/alternative in 2025

4. **Training Efficiency** (1-10)
   - 1-3: Basic training
   - 4-5: Gradient clipping, basic techniques
   - 6-7: Mixed precision (fp16)
   - 8: Advanced (bf16, fp8)
   - 9: Extreme efficiency techniques
   - 10: Best training efficiency in 2025

5. **Activation Functions** (1-10)
   - 1-2: Sigmoid/tanh
   - 3-4: ReLU
   - 5: Leaky ReLU, ELU
   - 6-7: GELU, Swish
   - 8: SwiGLU, GeGLU
   - 9: Advanced gated activations
   - 10: Best activation in 2025

6. **Normalization Techniques** (1-10)
   - 1: No normalization
   - 3: Batch normalization
   - 5-6: Layer normalization
   - 7: RMSNorm
   - 8: Advanced normalization
   - 9-10: Best normalization in 2025

7. **Learning Paradigm** (1-10)
   - 1-3: Basic supervised
   - 4-5: Self-supervised (early)
   - 6-7: Masked language modeling
   - 8: Advanced self-supervised
   - 9: Multi-modal, RLHF
   - 10: Best paradigm in 2025

8. **Regularization Techniques** (1-10)
   - 1-2: No regularization
   - 3-4: L2/weight decay
   - 5-6: Dropout
   - 7: Advanced dropout variants
   - 8-9: Novel regularization
   - 10: Best regularization in 2025

9. **Special Algorithms** (0-10)
   - 0: No special algorithms
   - Score based on how innovative/effective the algorithm is compared to 2025 best practices

10. **Overall Algorithm Quality** (1-10)
   - How close is this overall approach to 2025 state-of-the-art?
   - Consider all components together

Remember: Adam gets 6-7 whether used in 2014 or 2024. The scale is ABSOLUTE, not relative to the time.

Return ONLY a JSON object:
```json
{{
  "optimizer_quality": <score>,
  "architecture_quality": <score>,
  "attention_mechanism": <score>,
  "training_efficiency": <score>,
  "activation_functions": <score>,
  "normalization_techniques": <score>,
  "learning_paradigm": <score>,
  "regularization_techniques": <score>,
  "special_algorithms": <score>,
  "overall_algorithm_quality": <score>
}}
```
"""
    return prompt

def score_absolute_with_gemini(model, algo_text, system_name, year):
    """Get absolute scores from Gemini"""
    if pd.isna(algo_text) or not algo_text.strip():
        return None
    
    prompt = create_absolute_scoring_prompt(algo_text, system_name, year)
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract JSON
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        scores = json.loads(response_text.strip())
        return scores
    except Exception as e:
        print(f"Error processing {system_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Score algorithms on absolute scale')
    parser.add_argument('--api-key', required=True, help='Gemini API key')
    parser.add_argument('--start-row', type=int, default=0, help='Starting row')
    parser.add_argument('--end-row', type=int, default=10, help='Ending row')
    parser.add_argument('--delay', type=float, default=3.0, help='Delay between API calls')
    
    args = parser.parse_args()
    
    # Setup
    model = setup_gemini(args.api_key)
    df = pd.read_csv('papers/dataset.csv')
    
    # Initialize absolute score columns
    absolute_columns = [
        'optimizer_quality', 'architecture_quality', 'attention_mechanism',
        'training_efficiency', 'activation_functions', 'normalization_techniques',
        'learning_paradigm', 'regularization_techniques', 'special_algorithms',
        'overall_algorithm_quality'
    ]
    
    for col in absolute_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Process rows
    end_row = args.end_row or len(df)
    
    for idx in range(args.start_row, min(end_row, len(df))):
        row = df.iloc[idx]
        algo_text = row['algorithmic']
        system_name = row['System']
        year = row.get('Year', 2020)
        
        if pd.isna(algo_text) or not algo_text.strip():
            continue
        
        print(f"\nProcessing row {idx}: {system_name} ({year})")
        
        # Get absolute scores
        scores = score_absolute_with_gemini(model, algo_text, system_name, year)
        
        if scores:
            # Update dataframe
            for key, value in scores.items():
                df.at[idx, key] = value
            
            print(f"Absolute scores: {scores}")
            
            # Save after each row
            df.to_csv('papers/dataset_with_absolute_scores.csv', index=False)
        
        time.sleep(args.delay)
    
    print(f"\nSaved to dataset_with_absolute_scores.csv")

if __name__ == "__main__":
    main()