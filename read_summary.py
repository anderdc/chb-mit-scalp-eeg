# script reads the summary.txt from the data directories and parses the content into json
import os
import re
import json

def parse_summary_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    files_data = []
    current_file = {}

    for line in lines:
        line = line.strip()

        # Detect file name
        if line.startswith("File Name:"):
            if current_file:
                files_data.append(current_file)
            current_file = {
                "file_name": line.split(":")[1].strip(),
                "seizures": []
            }

        elif line.startswith("File Start Time:"):
            current_file["start_time"] = line.split(":")[1].strip()

        elif line.startswith("File End Time:"):
            current_file["end_time"] = line.split(":")[1].strip()

        elif line.startswith("Number of Seizures in File:"):
            current_file["num_seizures"] = int(line.split(":")[1].strip())

        elif line.startswith("Seizure Start Time:"):
            start = int(line.split(":")[1].replace('seconds', '').strip())
            current_file["seizures"].append({"start": start})

        elif line.startswith("Seizure End Time:"):
            end = int(line.split(":")[1].replace('seconds', '').strip())
            if current_file["seizures"]:
                current_file["seizures"][-1]["end"] = end

    if current_file:
        files_data.append(current_file)

    return files_data

def crawl_and_parse(base_dir):
    results = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("-summary.txt"):
                summary_path = os.path.join(root, file)
                print(f"Parsing {summary_path}")
                parsed = parse_summary_file(summary_path)
                results.extend(parsed)

    return results

if __name__ == "__main__":
    base_directory = "data"  # Change this to your root directory
    parsed_data = crawl_and_parse(base_directory)

    # Save as JSON
    with open("all_summary_data.json", "w+") as out:
        json.dump(parsed_data, out, indent=2)

    # Optional: Print CSV-style summary
    print("file_name,num_seizures,start_time,end_time,seizure_ranges")
    for f in parsed_data:
        seizure_ranges = ";".join([f"{s['start']}-{s['end']}" for s in f["seizures"]])
        print(f"{f['file_name']},{f['num_seizures']},{f['start_time']},{f['end_time']},{seizure_ranges}")
