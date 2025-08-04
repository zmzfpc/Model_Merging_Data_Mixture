#!/usr/bin/env python3
"""
Extract LINEAR experiment results from output file.
Extracts: Experiment number, Weight parameters, HumanEval/+ scores, MBPP/+ scores, 
and Code Summarization scores (Aggregate, Norm, Norm/Extract).
"""

import re
import csv
import sys
from pathlib import Path

def extract_linear_results(file_path):
    """Extract LINEAR experiment results from the output file."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Get the filename without path
    input_filename = Path(file_path).name
    
    results = []
    
    # Split content by experiment sections
    experiment_sections = re.split(r'LINEAR Experiment (\d+)/9: \d+', content)
    
    for i in range(1, len(experiment_sections), 2):
        experiment_num = experiment_sections[i]
        section_content = experiment_sections[i + 1] if i + 1 < len(experiment_sections) else ""
        
        if not section_content:
            continue
            
        # Extract weight parameters
        weight1_match = re.search(r'Model 1 - Weight: ([\d.]+)', section_content)
        weight2_match = re.search(r'Model 2 - Weight: ([\d.]+)', section_content)
        
        weight1 = weight1_match.group(1) if weight1_match else ""
        weight2 = weight2_match.group(1) if weight2_match else ""
        
        # Extract HumanEval results
        humaneval_base_match = re.search(r'humaneval \(base tests\)\s*\n\s*pass@1:\s*([\d.]+)', section_content)
        humaneval_plus_match = re.search(r'humaneval\+ \(base \+ extra tests\)\s*\n\s*pass@1:\s*([\d.]+)', section_content)
        
        humaneval_base = humaneval_base_match.group(1) if humaneval_base_match else ""
        humaneval_plus = humaneval_plus_match.group(1) if humaneval_plus_match else ""
        
        # Extract MBPP results
        mbpp_base_match = re.search(r'mbpp \(base tests\)\s*\n\s*pass@1:\s*([\d.]+)', section_content)
        mbpp_plus_match = re.search(r'mbpp\+ \(base \+ extra tests\)\s*\n\s*pass@1:\s*([\d.]+)', section_content)
        
        mbpp_base = mbpp_base_match.group(1) if mbpp_base_match else ""
        mbpp_plus = mbpp_plus_match.group(1) if mbpp_plus_match else ""
        
        # Extract Aggregate Scores
        aggregate_section = re.search(r'=== Aggregate Scores ===\s*\n(.*?)(?=\n=== Norm Scores ===)', section_content, re.DOTALL)
        agg_bleu4 = agg_chrf = agg_rouge = agg_meteor = ""
        if aggregate_section:
            agg_content = aggregate_section.group(1)
            agg_bleu4_match = re.search(r'BLEU-4\s*:\s*([\d.]+)', agg_content)
            agg_chrf_match = re.search(r'chrF\+\+\s*:\s*([\d.]+)', agg_content)
            agg_rouge_match = re.search(r'ROUGE-L\s*:\s*([\d.]+)', agg_content)
            agg_meteor_match = re.search(r'METEOR\s*:\s*([\d.]+)', agg_content)
            
            agg_bleu4 = agg_bleu4_match.group(1) if agg_bleu4_match else ""
            agg_chrf = agg_chrf_match.group(1) if agg_chrf_match else ""
            agg_rouge = agg_rouge_match.group(1) if agg_rouge_match else ""
            agg_meteor = agg_meteor_match.group(1) if agg_meteor_match else ""
        
        # Extract Norm Scores
        norm_section = re.search(r'=== Norm Scores ===\s*\n(.*?)(?=\n=== Norm/Extract Scores ===)', section_content, re.DOTALL)
        norm_bleu4 = norm_chrf = norm_rouge = norm_meteor = ""
        if norm_section:
            norm_content = norm_section.group(1)
            norm_bleu4_match = re.search(r'BLEU-4\s*:\s*([\d.]+)', norm_content)
            norm_chrf_match = re.search(r'chrF\+\+\s*:\s*([\d.]+)', norm_content)
            norm_rouge_match = re.search(r'ROUGE-L\s*:\s*([\d.]+)', norm_content)
            norm_meteor_match = re.search(r'METEOR\s*:\s*([\d.]+)', norm_content)
            
            norm_bleu4 = norm_bleu4_match.group(1) if norm_bleu4_match else ""
            norm_chrf = norm_chrf_match.group(1) if norm_chrf_match else ""
            norm_rouge = norm_rouge_match.group(1) if norm_rouge_match else ""
            norm_meteor = norm_meteor_match.group(1) if norm_meteor_match else ""
        
        # Extract Norm/Extract Scores
        norm_extract_section = re.search(r'=== Norm/Extract Scores ===\s*\n(.*?)(?=\n==========================================)', section_content, re.DOTALL)
        ne_bleu4 = ne_chrf = ne_rouge = ne_meteor = ""
        if norm_extract_section:
            ne_content = norm_extract_section.group(1)
            ne_bleu4_match = re.search(r'BLEU-4\s*:\s*([\d.]+)', ne_content)
            ne_chrf_match = re.search(r'chrF\+\+\s*:\s*([\d.]+)', ne_content)
            ne_rouge_match = re.search(r'ROUGE-L\s*:\s*([\d.]+)', ne_content)
            ne_meteor_match = re.search(r'METEOR\s*:\s*([\d.]+)', ne_content)
            
            ne_bleu4 = ne_bleu4_match.group(1) if ne_bleu4_match else ""
            ne_chrf = ne_chrf_match.group(1) if ne_chrf_match else ""
            ne_rouge = ne_rouge_match.group(1) if ne_rouge_match else ""
            ne_meteor = ne_meteor_match.group(1) if ne_meteor_match else ""
        
        # Store the results
        result = {
            'input_file': input_filename,
            'method': 'LINEAR',
            'experiment': int(experiment_num),
            'weight1': weight1,
            'weight2': weight2,
            'humaneval_base': humaneval_base,
            'humaneval_plus': humaneval_plus,
            'mbpp_base': mbpp_base,
            'mbpp_plus': mbpp_plus,
            'agg_bleu4': agg_bleu4,
            'agg_chrf': agg_chrf,
            'agg_rouge': agg_rouge,
            'agg_meteor': agg_meteor,
            'norm_bleu4': norm_bleu4,
            'norm_chrf': norm_chrf,
            'norm_rouge': norm_rouge,
            'norm_meteor': norm_meteor,
            'ne_bleu4': ne_bleu4,
            'ne_chrf': ne_chrf,
            'ne_rouge': ne_rouge,
            'ne_meteor': ne_meteor
        }
        
        results.append(result)
    
    return results

def write_to_csv(results, output_file):
    """Write results to CSV file."""
    
    fieldnames = [
        'input_file', 'method', 'experiment', 'weight1', 'weight2',
        'humaneval_base', 'humaneval_plus', 'mbpp_base', 'mbpp_plus',
        'agg_bleu4', 'agg_chrf', 'agg_rouge', 'agg_meteor',
        'norm_bleu4', 'norm_chrf', 'norm_rouge', 'norm_meteor',
        'ne_bleu4', 'ne_chrf', 'ne_rouge', 'ne_meteor'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_linear_results.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = f"csv_output/linear_experiment_results_{input_file.split(".")[0].split("/")[-1]}.csv"
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"Extracting LINEAR results from: {input_file}")
    results = extract_linear_results(input_file)
    
    if not results:
        print("No LINEAR experiment results found")
        sys.exit(1)
    
    print(f"Found {len(results)} experiments")
    write_to_csv(results, output_file)
    print(f"Results written to: {output_file}")
    
    # Print summary
    print("\nSummary:")
    for result in results:
        print(f"Experiment {result['experiment']}: W1={result['weight1']}, W2={result['weight2']}, "
              f"HumanEval={result['humaneval_base']}, MBPP={result['mbpp_base']}")

if __name__ == "__main__":
    main()
