#!/usr/bin/env python3
"""
Temporary script to extract abstracts from CSV and save as individual JSON files.
Each JSON file will contain: id, title, and title_abstract (title + abstract concatenated).
"""

import pandas as pd
import json
from pathlib import Path


def extract_abstracts_to_json(csv_path: str, output_dir: str):
    """
    Extract abstracts from CSV and save each as a separate JSON file.

    Args:
        csv_path: Path to the input CSV file
        output_dir: Directory where JSON files will be saved
    """
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Found {len(df)} abstracts in the CSV file")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")

    # Process each row
    success_count = 0
    error_count = 0

    for index, row in df.iterrows():
        try:
            # Create JSON object
            abstract_data = {
                "id": int(row['id']),
                "title": str(row['title']),
                "title_abstract": str(row['title']) + str(row['abstract'])
            }

            # Create filename
            filename = f"abstract_{row['id']}.json"
            filepath = output_path / filename

            # Write JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(abstract_data, f, indent=2, ensure_ascii=False)

            success_count += 1

            # Print progress every 10 files
            if success_count % 10 == 0:
                print(f"  Processed {success_count}/{len(df)} abstracts...")

        except Exception as e:
            print(f"Error processing row {index} (ID: {row.get('id', 'unknown')}): {e}")
            error_count += 1

    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total abstracts processed: {len(df)}")
    print(f"Successfully created: {success_count} JSON files")
    print(f"Errors encountered: {error_count}")
    print(f"Output location: {output_path.absolute()}")
    print("="*60)

    # List a few example files
    json_files = sorted(output_path.glob("*.json"))
    if json_files:
        print(f"\nExample files created:")
        for i, file in enumerate(json_files[:5]):
            print(f"  - {file.name}")
        if len(json_files) > 5:
            print(f"  ... and {len(json_files) - 5} more")


if __name__ == "__main__":
    # Configuration
    CSV_PATH = "uploads/rag.csv"
    OUTPUT_DIR = "uploads/rag_json"

    # Run extraction
    extract_abstracts_to_json(CSV_PATH, OUTPUT_DIR)
