#!/usr/bin/env python3
"""
Script to read Excel (.xlsx) files starting with 'PXD' and concatenate them into a single CSV file.
The PXD identifier will be added as a new column.
"""

import os
import glob
import pandas as pd
from pathlib import Path
import argparse
import numpy as np

ALLOWED_section_types = ['TITLE', 'ABSTRACT', 'INTRO', 'RESULTS', 'DISCUSS', 'FIG', 'METHODS', 'REF', 'SUPPL']

mapping = pd.read_csv('Controlled-vocab_mapping.csv')
print(mapping)

allowed_data_columns = pd.read_csv('/home/ianms/sandbox/merg_csv/allowed_columns.csv')
allowed_data_columns = allowed_data_columns['allowed_columns'].values
print(allowed_data_columns, len(allowed_data_columns))

def main():
    parser = argparse.ArgumentParser(
        description='Consolidate PXD Excel files into a single CSV file'
    )
    parser.add_argument(
        '-i', '--input',
        default='.',
        help='Input directory containing Excel files (default: current directory)'
    )
    parser.add_argument(
        '-o', '--output',
        default='TrainingSDRFs.csv',
        help='Output CSV filename (default: TrainingSDRFs.csv)'
    )
    
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.input, 'PXD*.tsv'))
    all_data = []

    for file in files:
        pxd_id = Path(file).stem  # Extract PXD identifier from filename
        pxd_id = pxd_id.split('_')[0]  # Remove extension if present
        df = pd.read_csv(file, sep='\t')
        df['PXD'] = pxd_id  # Add PXD_ID column to first position
        df = df[['PXD'] + [col for col in df.columns if col != 'PXD']]

        # map df columns to mapping where possible and if not present in mapping,
        # print a warning. If MappedTo == Unsure, drop the column
        # print(df)
        # print(df.columns, len(df.columns))
        mapped_df = pd.DataFrame()
        for col in df.columns:
            if col in mapping['Original'].values:
               
                new_col = mapping[mapping['Original'] == col]['MappedTo'].values[0]
                if new_col == 'Unsure':
                    print(f"Dropping column '{col}' as it is marked 'Unsure' in mapping.")
                    continue

                if new_col in mapped_df.columns:
                    # print(f"Warning: Mapped column name '{new_col}' already exists. Appending suffix to avoid duplication.")
                    suffix = 1
                    while f"{new_col}.{suffix}" in mapped_df.columns:
                        suffix += 1
                    new_col = f"{new_col}.{suffix}"

                print(f'Column "{col}" found in mapping -> Mapped to "{new_col}".')
                mapped_df[new_col] = df[col]

            else:
                print(f"Warning: Column '{col}' not found in mapping. It will be skipped.")
                quit()
        df = mapped_df
        print(df)
        # print(df.columns, len(df.columns))

        all_data.append(df)

    if all_data:
        consolidated_df = pd.concat(all_data, ignore_index=True)
        print(f"Total records consolidated: {len(consolidated_df)}")
        print(consolidated_df)

        # check if any columns are not in allowed_data_columns
        for col in consolidated_df.columns:
            if col not in allowed_data_columns:
                print(f"WARNING: Column {col} in consolidated data is not in sample submission")

        # add new columns from allowed_data_columns if missing
        for col in allowed_data_columns:
            if col not in consolidated_df.columns:
                print(f"Column {col} from sample submission is missing in consolidated data")
                consolidated_df[col] = 'Not Applicable'
        
        print(f"Columns in consolidated data:")
        for col in consolidated_df.columns:
            print(f" - {col}")
        print(f"Sample data:\n{consolidated_df.head()}")


        # add new column called ID which is just an index from 0 to n-1
        del consolidated_df['ID']
        consolidated_df.insert(0, 'ID', range(len(consolidated_df)))
        print(consolidated_df)


        # replace any NaN values with empty strings
        consolidated_df.fillna('Not Applicable', inplace=True)
        print(consolidated_df)

        consolidated_df.to_csv(args.output, index=False)
        print(f"Consolidated data written to {args.output}")


        # make a sample df by taking the first 10 rows and writing to Sample_submission.csv
        sample_df = consolidated_df.head(10)
        sample_df.to_csv('sample_submission.csv', index=False)
        print("sample submission data written to Sample_submission.csv")

    else:
        print("No valid PXD files found.")

    print("Done.")
if __name__ == '__main__':
    main()
