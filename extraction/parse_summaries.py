import os
import re
import json
import argparse

def parse_summary(filepath: str) -> dict:
    """
    Parses a CHB-MIT .summary.txt file to extract seizure times for each EEG file listed.

    Args:
        filepath (str): Path to a -summary.txt file.

    Returns:
        dict: A dictionary with a single key "seizures", whose value is a list of dictionaries.
              Each dictionary represents a file and includes:
                - file_name (str)
                - seizure_count (int)
                - seizure_times (list of dicts with "start" and "end" times)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    response = {
        "seizures": list()
    }

    current_file = None

    for line in lines:
        line = line.strip()

        if line.startswith("File Name:"):
            # Push the previous file before starting a new one
            if current_file:
                response["seizures"].append(current_file)
            current_file = {
                "file_name": line.split(":")[1].strip(),
                "seizure_times": [],
            }

        elif line.startswith("Number of Seizures in File:") and current_file:
            current_file["seizure_count"] = int(line.split(":")[1].strip())

        elif "Seizure" in line and "Start Time" in line and current_file:
            start = int(line.split(":")[1].replace('seconds', '').strip())
            current_file["seizure_times"].append({"start": start})

        elif "Seizure" in line and "End Time" in line and current_file:
            end = int(line.split(":")[1].replace('seconds', '').strip())
            if current_file["seizure_times"]:
                current_file["seizure_times"][-1]["end"] = end

    # Append the last file's data
    if current_file:
        response["seizures"].append(current_file)

    return response


def filter_seizures(parsed_summary: dict) -> list:
    """
    Filters out files that contain no seizures from the parsed summary data.

    Args:
        parsed_summary (dict): Output of `parse_summary()`, containing a list of seizure records.

    Returns:
        list: A list of file-level seizure records where seizure_count > 0.
    """
    seizures = parsed_summary["seizures"]
    filtered_seizures = list()
    for seizure in seizures:
        if seizure["seizure_count"] > 0:
            filtered_seizures.append(seizure)

    return filtered_seizures


def crawl_and_parse(base_dir: str) -> list:
    """
    Recursively searches for -summary.txt files under a given directory,
    parses each one, and extracts all seizure times.

    Args:
        base_dir (str): Root directory containing CHB-MIT patient folders.

    Returns:
        list: Combined list of seizure entries from all parsed summary files.
    """
    results = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("-summary.txt"):
                summary_path = os.path.join(root, file)
                print(f"Parsing {summary_path.split('/')[-1]}")
                parsed = parse_summary(summary_path)
                results.extend(filter_seizures(parsed))

    return results


if __name__ == "__main__":
    '''
    Example usage (run from project root):

        python -m extraction.parse_summaries --data-path ./data

    This script will:
    - Recursively find all -summary.txt files under the specified directory
    - Parse them for seizure information
    - Save all results as a JSON file at the root of the dataset directory
    '''
    parser = argparse.ArgumentParser(description="Summary Analyzer")
    parser.add_argument(
        "--data-path",
        type=str,
        default='./data',
        help="Path to the root directory containing CHB-MIT patient folders"
    )
    args = parser.parse_args()

    parsed_data = crawl_and_parse(args.data_path)

    # Save as JSON at the root of dataset directory
    with open(f"{args.data_path}/all_summary_data.json", "w+") as out:
        json.dump(parsed_data, out, indent=2)

    print(f"Done. Saved to file {args.data_path}/all_summary_data.json")
