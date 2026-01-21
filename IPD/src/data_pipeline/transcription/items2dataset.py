import json
import os
import pandas as pd
from tqdm import tqdm
import argparse

def process_segments(segments):
    """
    Processes segments to calculate the average score and create a cleaned list of segments.
    """
    processed_segments = []
    full_transcript = []
    if not segments:
        return [], ""

    for segment in segments:
        words = segment.get("words", [])
        if words:
            # Filter out words with None scores before calculating the average
            scores = [word.get("score") for word in words if word.get("score") is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
            else:
                avg_score = 0.0  # or some other default value
        else:
            avg_score = 0.0

        processed_segments.append({
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": segment.get("text", "").strip(),
            "avg_score": avg_score
        })
        full_transcript.append(segment.get("text", "").strip())
    
    return processed_segments, " ".join(full_transcript)

def segments_to_words(segments):
    """
    Extracts all words from segments with their start time, end time, and score.
    Returns a list of word dictionaries.
    """
    all_words = []
    
    if not segments:
        return all_words
    
    for segment in segments:
        words = segment.get("words", [])
        for word_info in words:
            all_words.append({
                "word": word_info.get("word", ""),
                "start": word_info.get("start"),
                "end": word_info.get("end"),
                "score": word_info.get("score")
            })
    
    return all_words

def create_dataset(raw_data_path, output_dir, formats=['parquet']):
    """
    Walks through the raw_data_path, processes JSON files for each podcast,
    and creates a separate dataset for each in the specified format(s).
    
    Args:
        raw_data_path: Path to the raw JSON data
        output_dir: Directory to save the datasets
        formats: List of output formats ('parquet', 'csv')
    """
    # Create output directories if they don't exist
    parquet_output_dir = None
    csv_output_dir = None
    
    if 'parquet' in formats:
        parquet_output_dir = os.path.join(output_dir, "parquet")
        os.makedirs(parquet_output_dir, exist_ok=True)
    
    if 'csv' in formats:
        csv_output_dir = os.path.join(output_dir, "CSV")
        os.makedirs(csv_output_dir, exist_ok=True)

    podcast_folders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]

    for podcast_name in tqdm(podcast_folders, desc="Processing podcasts"):
        data = []
        podcast_folder_path = os.path.join(raw_data_path, podcast_name)
        
        for item in os.listdir(podcast_folder_path):
            if item.endswith(".json"):
                file_path = os.path.join(podcast_folder_path, item)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        episode_data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Skipping corrupted JSON file: {file_path}")
                        continue

                metadata = episode_data.get("metadata", {})
                result = episode_data.get("result", {})
                segments = result.get("segments", [])
                
                processed_segments, transcript = process_segments(segments)
                all_words = segments_to_words(segments)

                data.append({
                    "podcast_title": podcast_name,
                    "episode_title": metadata.get("title"),
                    "description": metadata.get("description"),
                    "date": metadata.get("published") or metadata.get("date"),
                    "audio_url": metadata.get("audio_url"),
                    "segments": processed_segments,
                    "transcript": transcript,
                    "words": all_words
                })
        
        if not data:
            continue

        df = pd.DataFrame(data)
        created_files = []

        # Save in Parquet format if requested
        if 'parquet' in formats and parquet_output_dir:
            output_parquet_path = os.path.join(parquet_output_dir, f"{podcast_name}.parquet")
            df.to_parquet(output_parquet_path, index=False)
            created_files.append(output_parquet_path)
        
        # Save in CSV format if requested
        if 'csv' in formats and csv_output_dir:
            output_csv_path = os.path.join(csv_output_dir, f"{podcast_name}.csv")
            # For CSV, convert the 'segments' and 'words' list of dicts to JSON strings
            df_for_csv = df.copy()
            df_for_csv['segments'] = df_for_csv['segments'].apply(json.dumps)
            df_for_csv['words'] = df_for_csv['words'].apply(json.dumps)
            df_for_csv.to_csv(output_csv_path, index=False)
            created_files.append(output_csv_path)

        if created_files:
            print(f"Datasets for '{podcast_name}' created: {', '.join(created_files)}")
        else:
            print(f"No datasets created for '{podcast_name}' - no valid formats specified")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process podcast JSON data into datasets."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "raw_data"),
        help="The directory where the raw data is located. Defaults to 'raw_data' folder in the same directory as this script.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "dataset"),
        help="The directory to store the processed datasets. Defaults to 'dataset' directory from the root of the project.",
    )
    parser.add_argument(
        "-f",
        "--formats",
        nargs='+',
        choices=['parquet', 'csv'],
        default=['parquet'],
        help="Output formats to generate. Choose from 'parquet' and/or 'csv'. Defaults to 'parquet' only.",
    )
    args = parser.parse_args()
    
    create_dataset(args.input_dir, args.output_dir, args.formats)
