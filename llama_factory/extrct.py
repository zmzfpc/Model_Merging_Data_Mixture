#!/usr/bin/env python3
import pandas as pd
import os

def recreate_combined_csv():
    """Recreate the combined ct vs 4o correlations CSV file"""
    
    # Define the input files and their corresponding dataset names
    files_data = [
        {
            'file': 'vis/gap_cor_dsc7_{0.3}/layer_by_layer_correlations.csv',
            'dataset': 'dsc7',
            'ct_model': 'sft_ct_1e6',
            '4o_model': 'sft_4o_sol_5e6'
        },
        {
            'file': 'vis/gap_cor_dsc13_{0.3}/layer_by_layer_correlations.csv', 
            'dataset': 'dsc13',
            'ct_model': 'sft_ct_5e6',
            '4o_model': 'sft_4o_sol_1e5'
        },
        {
            'file': 'vis/gap_cor_qwc7_{0.3}/layer_by_layer_correlations.csv',
            'dataset': 'qwc7', 
            'ct_model': 'sft_ct_1e6',
            '4o_model': 'sft_4o_sol_5e6'
        },
        {
            'file': 'vis/gap_cor_qwc15_{0.3}/layer_by_layer_correlations.csv',
            'dataset': 'qwc15', 
            'ct_model': 'sft_ct_5e6',
            '4o_model': 'sft_4o_sol_1e-5'
        }
    ]
    
    combined_data = []
    
    for file_info in files_data:
        file_path = file_info['file']
        dataset_name = file_info['dataset']
        ct_model = file_info['ct_model']
        fo_model = file_info['4o_model']
        
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Filter for ct vs 4o comparisons
            ct_vs_4o = df[(df['model_1'] == ct_model) & (df['model_2'] == fo_model)].copy()
            
            # Add dataset identifier
            ct_vs_4o['dataset'] = dataset_name
            
            # Add to combined data
            combined_data.append(ct_vs_4o)
            
            print(f"  Found {len(ct_vs_4o)} ct vs 4o correlation records")
        else:
            print(f"Warning: {file_path} not found!")
    
    if combined_data:
        # Combine all dataframes
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Reorder columns to put dataset first after layer_num
        column_order = ['layer_num', 'dataset', 'model_1', 'model_2', 'correlation', 
                       'p_value', 'n_samples', 'significant', 'significance_level']
        combined_df = combined_df[column_order]
        
        # Sort by dataset and layer_num
        combined_df = combined_df.sort_values(['dataset', 'layer_num'])
        
        # Save to output file
        output_file = 'combined_ct_vs_4o_correlations.csv'
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Combined data saved to {output_file}")
        print(f"Total records: {len(combined_df)}")
        print(f"Datasets included: {sorted(combined_df['dataset'].unique())}")
        
        return True
    else:
        print("❌ No data found to combine!")
        return False

if __name__ == "__main__":
    success = recreate_combined_csv()
    if success:
        print("\n🎯 Ready to run plotting script!")
    else:
        print("\n⚠️ Failed to create combined CSV")