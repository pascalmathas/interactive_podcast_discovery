from __future__ import annotations
from abc import abstractmethod
import whisperx
import gc
import argparse
import torch
import os
import json
import time
from typing import Dict, Any, Optional
from email.utils import parsedate_to_datetime
from datetime import datetime
# --------------------------------------------------------------------
# HPC / SLURM runtime tweaks
# --------------------------------------------------------------------
# 1. Prevent unnecessary internet traffic on compute nodes (HF telemetry)
import os as _os  # alias for local scope isolation
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# 2. Respect assigned CPU cores (SLURM_CPUS_PER_TASK) for Torch
if "SLURM_CPUS_PER_TASK" in _os.environ:
    try:
        import torch as _torch
        _torch.set_num_threads(int(_os.environ["SLURM_CPUS_PER_TASK"]))
        print(f"torch.set_num_threads({_os.environ['SLURM_CPUS_PER_TASK']}) set based on SLURM_CPUS_PER_TASK")
    except Exception as exc:
        print(f"Warning: could not apply SLURM_CPUS_PER_TASK: {exc}")
# --------------------------------------------------------------------

# Core deps
import feedparser
# Local modules
from rss_utils import download_audio_entry as _download_entry, sanitize_for_path as _sanitize_for_path
from subtitles import save_transcripts

def _process_audio_file(
    audio_path: str,
    model: Any,
    device: str,
    batch_size: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Transcribes and aligns a single audio file."""
    print(f"\n=== Processing {os.path.basename(audio_path)} ===")

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    # Align
    model_a, align_meta = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        align_meta,
        audio,
        device,
        return_char_alignments=False,
    )

    # Cleanup alignment objects
    del model_a, align_meta, audio
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return result


def _process_rss_feed(
    rss_url: str,
    model: Any,
    device: str,
    batch_size: int,
    output_dir: Optional[str] = None,
    feed_obj: Optional[feedparser.FeedParserDict] = None,
    podcast_feed_json_title: str | None = None,
    json_processor: Optional["JsonProcessor"] = None,
) -> None:
    """Processes a single RSS feed: downloads all episodes and transcribes them."""
    print(f"\n=== Processing RSS feed: {rss_url} ===")
    
    feed = feed_obj if feed_obj else feedparser.parse(rss_url)
    if feed.bozo:
        print(f"Error parsing RSS feed {rss_url}: {feed.bozo_exception}")
        return

    podcast_title = podcast_feed_json_title if podcast_feed_json_title else feed.feed.get("title", "podcast_name_not_found")  # type: ignore
    base_root = output_dir if output_dir else "raw_data"
    podcast_dir = os.path.join(base_root, _sanitize_for_path(podcast_title))
    os.makedirs(podcast_dir, exist_ok=True)
    
    print(f"Podcast: {podcast_title} ({len(feed.entries)} episodes)")

    if not feed.entries:
        print("No episodes found in RSS feed.")
        return

    # Filter episodes that have already been processed
    entries_to_process = feed.entries
    if json_processor and podcast_feed_json_title:
        try:
            podcast_item, _ = json_processor.get_podcast_feed_object_by_title(podcast_feed_json_title, return_index=True)
            info = podcast_item.get("additional_transcription_info", {})
            
            if info and info.get("last_processed_episode_pub_date") != "unknown":
                print("Previously processed episodes found, filtering...")
                
                first_date_str = info["first_processed_episode_pub_date"]
                last_date_str = info["last_processed_episode_pub_date"]

                try:
                    # Convert saved dates to datetime objects
                    first_processed_dt = parsedate_to_datetime(first_date_str)
                    last_processed_dt = parsedate_to_datetime(last_date_str)

                    # Ensure the most recent date is used as 'end'
                    start_range = min(first_processed_dt, last_processed_dt)
                    end_range = max(first_processed_dt, last_processed_dt)

                    filtered_entries = []
                    for entry in feed.entries:
                        # Get the publication date of the episode
                        entry_pub_date = entry.get("published_parsed")
                        if not entry_pub_date or not isinstance(entry_pub_date, time.struct_time):
                            # Add if we can't parse the date, for safety
                            filtered_entries.append(entry)
                            continue

                        entry_dt = datetime.fromtimestamp(time.mktime(entry_pub_date))
                        entry_dt = entry_dt.replace(tzinfo=first_processed_dt.tzinfo) # Make timezone-aware

                        # Do not add the episode if the date falls within the processed range
                        if not (start_range <= entry_dt <= end_range):
                            filtered_entries.append(entry)
                    
                    entries_to_process = filtered_entries
                    print(f"Filtering completed: {len(entries_to_process)} of {len(feed.entries)} episodes will be processed.")

                except (ValueError, TypeError) as e:
                    print(f"Warning: could not parse dates ('{first_date_str}', '{last_date_str}'). Processing all episodes. Error: {e}")

        except ValueError:
            # Podcast not yet in JSON, process everything
            pass

    for i, entry in enumerate(entries_to_process, 1):
        print(f"\n--- Processing episode {i}/{len(entries_to_process)}: {entry.get('title', 'No title')} ---")
        
        episode_info = _download_entry(entry, rss_url, podcast_dir)
        if episode_info is None:
            print("No audio found, skipped.")
            continue

        audio_file = episode_info["filepath"]
        metadata = episode_info["metadata"]

        result = _process_audio_file(
            audio_path=audio_file,
            model=model,
            device=device,
            batch_size=batch_size,
            metadata=metadata,
        )

        transcript_dir = podcast_dir
        os.makedirs(transcript_dir, exist_ok=True)
        base_filename = os.path.join(transcript_dir, os.path.splitext(os.path.basename(audio_file))[0])
        save_transcripts(result, base_filename, metadata=metadata)

        # Update progress in the JSON file
        if json_processor and podcast_feed_json_title:
            pub_date = str(entry.get("published", "unknown date"))
            json_processor.update_episode_progress(podcast_feed_json_title, pub_date)

        try:
            os.remove(audio_file)
            print(f"Temporary file removed: {audio_file}")
        except OSError as exc:
            print(f"Warning: could not remove temporary file {audio_file}: {exc}")


def generate_transcript(
    input_path_or_url,
    device_arg="auto",
    compute_type_arg="auto",
    model_name="large-v3",
    model_dir=os.environ.get("WHISPER_MODEL_DIR", "./model_checkpoints"),
    batch_size_param=16,
    local_only=False,
    output_dir=None,
):
    """Generates transcriptions with WhisperX.

    * If *input_path_or_url* is a **local audio file**, only that file is processed.
    * If it points to an **RSS feed** (http/https), **all** episodes with audio
      in that feed are downloaded and processed sequentially.
    * If it's a **JSON file** with podcast feeds, all feeds in the file are processed.
      The JSON file must contain a list with objects that include 'rss_feed_url' and 'podcast_title'.

    All files are saved by default in the ``raw_data/`` folder:
    - RSS feeds: ``raw_data/<podcast_title>/``
    - Local files: ``raw_data/local_files/``
    
    For each episode, the transcription is saved as ``<basename>.json``.
    For RSS feeds, the JSON file also contains the episode metadata.
    """
    # ------------------------------------------------------------------
    # 1. Determine execution device and compute type
    # ------------------------------------------------------------------
    device = device_arg
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Device 'cuda' was requested but is not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Device 'mps' was requested but is not available.")

    # faster-whisper (for transcription) doesn't support mps, so fall back to cpu
    transcription_device = "cpu" if device == "mps" else device

    print(f"Using device: {device}")
    if device == "mps":
        print("Note: 'cpu' is used for transcription; alignment runs on 'mps'.")

    compute_type = compute_type_arg
    if compute_type == "auto":
        compute_type = "int8" if transcription_device == "cpu" else "float16"
    print(f"Using compute type: {compute_type}")

    abs_model_dir = os.path.abspath(model_dir)
    if local_only:
        expected_dir = os.path.join(abs_model_dir, model_name)
        if not os.path.exists(expected_dir):
            raise FileNotFoundError(
                f"local_only=True, but model files not found at {expected_dir}."
            )

    # ------------------------------------------------------------------
    # 2. Load the model (once for all files)
    # ------------------------------------------------------------------
    print("Loading WhisperX transcription model...")
    model = whisperx.load_model(
        model_name,
        transcription_device,
        compute_type=compute_type,
        download_root=abs_model_dir,
        local_files_only=local_only,
    )

    try:
        # ------------------------------------------------------------------
        # 3. Determine processing mode (JSON, RSS feed, or single file)
        # ------------------------------------------------------------------
        is_rss_feed = input_path_or_url.startswith("http://") or input_path_or_url.startswith("https://")
        is_json_file = (
            not is_rss_feed 
            and input_path_or_url.lower().endswith(".json") 
            and os.path.exists(input_path_or_url)
        )

        if is_json_file:
            # --------------------------- JSON with podcast feeds -----------
            print(f"JSON file detected: {input_path_or_url}")
            
            try:
                json_processor = JsonProcessor(input_path_or_url)
                podcast_feeds = json_processor.get_podcast_feeds()
            except ValueError as e:
                print(f"Error validating or loading {input_path_or_url}: {e}")
                return
            
            print(f"Found {len(podcast_feeds)} podcast feeds to process.")
            
            for i, podcast_info in enumerate(podcast_feeds):
                rss_url = podcast_info.get("rss_feed_url")
                podcast_title = podcast_info.get("podcast_title")

                if not rss_url or not podcast_title:
                    print(f"Skipped invalid entry at index {i}: missing 'rss_feed_url' or 'podcast_title'.")
                    continue
                
                if podcast_info.get("transcribed_status", False):
                    print(f"Podcast '{podcast_title}' is already marked as transcribed, skipped.")
                    continue
                
                # Parse feed to check metadata and initialize
                feed = feedparser.parse(rss_url)
                if feed.bozo:
                    print(f"Error parsing RSS feed {rss_url} for '{podcast_title}': {feed.bozo_exception}")
                    print("This podcast will be skipped.")
                    continue

                # Initialize or update 'additional_transcription_info' to have the latest status.
                # This function will not reset progress if it already exists.
                total_episodes = len(feed.entries)
                json_processor.init_additional_transcription_info(
                    podcast_title=podcast_title, 
                    total_episodes=total_episodes
                )
                
                print(f"\n{'='*60}")
                print(f"Processing podcast {i+1}/{len(podcast_feeds)}: {podcast_title}")
                print(f"{'='*60}")
                
                try:
                    _process_rss_feed(
                        rss_url=rss_url,
                        model=model,
                        device=device,
                        batch_size=batch_size_param,
                        output_dir=output_dir,
                        feed_obj=feed,
                        podcast_feed_json_title=podcast_title,
                        json_processor=json_processor,
                    )
                except Exception as exc:
                    print(f"Error processing '{podcast_title}' ({rss_url}): {exc}")
                    print("Continuing with next podcast...")
                    continue
                

        elif is_rss_feed:
            # --------------------------- RSS feed -------------------------#
            _process_rss_feed(
                rss_url=input_path_or_url,
                model=model,
                device=device,
                batch_size=batch_size_param,
                output_dir=output_dir,
            )

        else:
            # --------------------------- Single file ----------------------#
            if not os.path.exists(input_path_or_url):
                raise FileNotFoundError(f"Error: File not found at {input_path_or_url}")

            result = _process_audio_file(
                audio_path=input_path_or_url,
                model=model,
                device=device,
                batch_size=batch_size_param,
            )

            # Determine output path and save transcriptions
            if output_dir:
                podcast_dir = output_dir
            else:
                # For local files: place in raw_data/local_files
                podcast_dir = os.path.join("raw_data", "local_files")
            os.makedirs(podcast_dir, exist_ok=True)
            base_filename = os.path.join(podcast_dir, os.path.splitext(os.path.basename(input_path_or_url))[0])
            save_transcripts(result, base_filename, metadata=None)

    finally:
        # ------------------------------------------------------------------
        # 4. Clean up main model and memory
        # ------------------------------------------------------------------
        del model
        gc.collect()
        if transcription_device == "cuda":
            torch.cuda.empty_cache()

class JsonProcessor:
    """
    This class handles all the read and edit work of the JSON file with podcast feeds.
    """
    def __init__(self, json_file_path):
        self.json_file_path = JsonProcessor.check_json_schema(json_file_path)
        self.podcast_feeds = self._load_podcast_feeds()

    def get_podcast_feeds(self):
        return self.podcast_feeds
    
    def get_total_episodes(self, feed_object: feedparser.FeedParserDict) -> int:
        return len(feed_object.entries)

    def get_podcast_feed_object_by_title(self, podcast_title: str, return_index: bool = False) -> tuple[dict, int] | dict:
        for index, podcast_item in enumerate(self.podcast_feeds):
            if podcast_item.get("podcast_title") == podcast_title:
                if return_index:
                    return podcast_item, index
                else:
                    return podcast_item
        raise ValueError(f"Podcast with title '{podcast_title}' not found.")

    def get_episode_entry_by_title(self, episode_title: str, feed_object: feedparser.FeedParserDict) -> feedparser.FeedParserDict:
        for episode in feed_object.entries:
            if episode.get("title") == episode_title:
                return episode
        raise ValueError(f"Episode with title '{episode_title}' in feed with title '{feed_object.feed.get('title', 'unknown')}' not found.")  # type: ignore
    
    def get_podcast_feed_object_by_index(self, index: int):
        if not (0 <= index < len(self.podcast_feeds)):
            raise IndexError(f"Item index {index} is out of bounds.")
        return self.podcast_feeds[index]
    
    def has_additional_transcription_info(self, podcast_title: str) -> bool:
        podcast_item, _ = self.get_podcast_feed_object_by_title(podcast_title, return_index=True)
        return "additional_transcription_info" in podcast_item

    def init_additional_transcription_info(self, podcast_title: str, total_episodes: int = 0):
        podcast_item, podcast_item_index = self.get_podcast_feed_object_by_title(podcast_title, return_index=True)
        if "additional_transcription_info" in podcast_item:
            # Update total episodes and adjust status if necessary, preserving progress
            info = podcast_item["additional_transcription_info"]
            info["n_total_episodes"] = total_episodes

            # If it was completed but new episodes are available, reset status
            if info.get("status") == "completed" and info.get("n_processed_episodes", 0) < total_episodes:
                info["status"] = "in_progress"
                podcast_item["transcribed_status"] = False  # Allow reprocessing
                print(f"New episodes found for '{podcast_title}'. Status updated to 'in_progress'.")

            elif info.get("status") == "in_progress" and info.get("n_processed_episodes", 0) < total_episodes:
                info["n_total_episodes"] = total_episodes
                print(f"New episodes found for '{podcast_title}'. Progress updated.")

        else:
            podcast_item["additional_transcription_info"] = {
                "status": "pending",
                "n_total_episodes": total_episodes,
                "n_processed_episodes": 0,
                "first_processed_episode_pub_date": "unknown",
                "last_processed_episode_pub_date": "unknown",
            }
        self.podcast_feeds[podcast_item_index] = podcast_item
        self._write_to_json()
        print(f"Initialized/updated additional transcription info for podcast '{podcast_title}'.")

    def update_episode_progress(self, podcast_title: str, episode_pub_date: str):
        """Updates transcription progress after processing one episode."""
        podcast_item, index = self.get_podcast_feed_object_by_title(podcast_title, return_index=True)
        info = podcast_item.get("additional_transcription_info")

        if not info:
            print(f"Warning: cannot update progress, 'additional_transcription_info' missing for '{podcast_title}'.")
            return

        # Update progress
        info["n_processed_episodes"] += 1
        info["last_processed_episode_pub_date"] = episode_pub_date
        if info["first_processed_episode_pub_date"] == "unknown":
            info["first_processed_episode_pub_date"] = episode_pub_date

        # Update status
        if info["status"] == "pending":
            info["status"] = "in_progress"

        # Check for completion
        if info["n_processed_episodes"] >= info["n_total_episodes"]:
            info["status"] = "completed"
            podcast_item["transcribed_status"] = True
            print(f"Podcast '{podcast_title}' fully processed.")
        
        self.podcast_feeds[index] = podcast_item
        self._write_to_json()
        
        processed = info['n_processed_episodes']
        total = info['n_total_episodes']
        print(f"Progress for '{podcast_title}' updated: {processed}/{total} episodes processed.")

    def _write_to_json(self):
        with open(self.json_file_path, "w", encoding="utf-8") as fh:
            json.dump(self.podcast_feeds, fh, indent=2, ensure_ascii=False)

    def _load_podcast_feeds(self):
        with open(self.json_file_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    
    def _update_transcribed_status(
        self, item_index: int, podcast_title: str, transcribed_status: bool = False
    ):
        """
        Update or add the transcribed_status to the podcast object.
        When the provided podcast_title doesn't match the podcast_title of the podcast object at the given index,
        an error is thrown.
        When the transcribed_status doesn't exist, it's added with the provided value (default = False).
        When the transcribed_status exists, it's updated with the provided value.
        """
        if not (0 <= item_index < len(self.podcast_feeds)):
            raise IndexError(f"Item index {item_index} is out of bounds.")

        podcast_item = self.podcast_feeds[item_index]

        if podcast_item.get("podcast_title") != podcast_title:
            raise ValueError(
                f"Podcast title mismatch for index {item_index}. "
                f"Expected '{podcast_item.get('podcast_title')}', but got '{podcast_title}'."
            )

        podcast_item["transcribed_status"] = transcribed_status
        self.podcast_feeds[item_index] = podcast_item

        # Write changes back to the JSON file
        with open(self.json_file_path, "w", encoding="utf-8") as fh:
            json.dump(self.podcast_feeds, fh, indent=2, ensure_ascii=False)

    @staticmethod
    def check_json_schema(path: str):
        """
        Checks if the JSON file conforms to the schema.
        When the transcribed key doesn't exist, it's added with the value False.
        Returns path string if it conforms to the schema, throws error if it doesn't.

        Schema:
        Main level is a list containing dictionaries with the following keys:
        - podcast_title: str
        - rss_feed_url: str (URL of the RSS feed)
        - transcribed_status: bool (whether the podcast has been transcribed)
        Optional:
        - additional_transcription_info: dictionary with the following keys:
            - status: Enum: "pending", "in_progress", "completed", "failed"
            - n_total_episodes: int (total number of episodes in the feed)
            - n_processed_episodes: int (number of episodes already processed)
            - first_processed_episode_pub_date: str (pub_date of first processed episode)
            - last_processed_episode_pub_date: str (pub_date of last processed episode)

        Example:
        [
            {
                "podcast_title": "Example podcast name",
                "rss_feed_url": "https://example.com/rss",
                "transcribed_status": false,
                "additional_transcription_info": {
                    "status": "pending",
                    "n_total_episodes": 10,
                    "n_processed_episodes": 2,
                    "first_processed_episode_pub_date": "Tue, 17 Jun 2025 14:00:00 -0000",
                    "last_processed_episode_pub_date": "Tue, 10 Jun 2025 14:00:00 -0000"
                    }
            }
        ]
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"JSON file not found at path: {path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {path}")

        if not isinstance(data, list):
            raise ValueError("JSON root must be a list of podcast objects.")

        modified = False
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item at index {i} is not a dictionary.")

            # Check for required keys
            if "podcast_title" not in item or not isinstance(item["podcast_title"], str):
                raise ValueError(f"Item at index {i} is missing or has an invalid 'podcast_title'.")
            if "rss_feed_url" not in item or not isinstance(item["rss_feed_url"], str):
                raise ValueError(f"Item at index {i} is missing or has an invalid 'rss_feed_url'.")

            # Check and add 'transcribed_status' if missing
            if "transcribed_status" not in item:
                item["transcribed_status"] = False
                modified = True
                print(f'Added the "transcribed_status" key with default value False to the item with podcast title: {item["podcast_title"]}.')
            elif not isinstance(item["transcribed_status"], bool):
                raise ValueError(f"Item at index {i} has a non-boolean 'transcribed_status' value.")

            # Optional: Check 'additional_transcription_info'
            if "additional_transcription_info" in item:
                status_obj = item["additional_transcription_info"]
                if not isinstance(status_obj, dict):
                    raise ValueError(f"Item at index {i} has an invalid 'additional_transcription_info' (must be a dictionary).")

                required_keys = {
                    "status": str,
                    "n_total_episodes": int,
                    "n_processed_episodes": int,
                    "first_processed_episode_pub_date": str,
                    "last_processed_episode_pub_date": str,
                }
                for key, key_type in required_keys.items():
                    if key not in status_obj or not isinstance(status_obj[key], key_type):
                        raise ValueError(f"Item at index {i} has an invalid or missing '{key}' in 'additional_transcription_info'.")

                valid_statuses = ["pending", "in_progress", "completed", "failed"]
                if status_obj["status"] not in valid_statuses:
                    raise ValueError(f"Item at index {i} has an invalid 'status' value in 'additional_transcription_info'. Must be one of {valid_statuses}.")

        if modified:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"JSON file '{path}' has been updated with default 'transcribed_status' values.")
        else:
            print(f"JSON file '{path}' is all good. :)")

        return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Transcribe an audio file, podcast from an RSS feed, or multiple podcasts from a JSON file.
        Saves the transcription in JSON format in the 'raw_data/' folder.

        Examples:
        - For a local file (saved in raw_data/local_files/):
          python file2transcript.py my_audio.mp3

        - For a podcast feed (saved in raw_data/<podcast_name>/):
          python file2transcript.py https://feeds.simplecast.com/your-podcast-feed

        - For a JSON file with podcast feeds (each podcast in its own subfolder):
          python file2transcript.py podcast_feeds.json
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a local audio file, URL of an RSS podcast feed, or JSON file with podcast feeds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device for computations (e.g. 'cuda', 'cpu', 'mps'). 'auto' selects the best available device.",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="auto",
        choices=["auto", "float16", "int8", "float32"],
        help="Compute type for transcription. 'auto' selects the best option for the device.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="large-v3",
        help="Name of the Whisper model (e.g. 'large-v3', 'medium').",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("WHISPER_MODEL_DIR", "./model_checkpoints"),
        help="Directory where WhisperX looks for/downloads model checkpoints. Can also be set via WHISPER_MODEL_DIR env var.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for WhisperX transcription (reduce for GPU memory errors).",
    )
    parser.add_argument(
        "--local_only",
        action="store_true",
        help="Stop immediately if model files are not locally available (useful on offline HPC nodes).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where transcription files are saved. Defaults to 'raw_data/' folder with podcast-specific subfolders.",
    )
    args = parser.parse_args()

    if "SLURM_JOB_ID" in os.environ:
        print(
            f"Detected SLURM environment. Job ID: {os.environ.get('SLURM_JOB_ID')}, Node List: {os.environ.get('SLURM_NODELIST')}"
        )

    generate_transcript(
        input_path_or_url=args.input,
        device_arg=args.device,
        compute_type_arg=args.compute_type,
        model_name=args.model_name,
        model_dir=args.model_dir,
        batch_size_param=args.batch_size,
        local_only=args.local_only,
        output_dir=args.output_dir,
    )