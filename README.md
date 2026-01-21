# Interactive Podcast Discovery

Interactive Podcast Discovery (IPD) is a visual analytics system that transforms raw podcast audio into an interactive semantic landscape. Our system combines LLM-based transcript segmentation with an interactive visualization to enable podcast discovery. Users can explore content through a 2D interactive cluster map with semantic zoom, complemented by a LLM interface. This dual-interface approach addresses the challenge of discovering relevant content within audio podcasts.

---

## Run Dash Visualization

### Some initial info

**Two API keys are required:**

- OpenAI API key
- Google API key

**On the first startup, the whole BERTopic pipeline will be executed, this will take 30 minutes**

**On subsequent startups, it will use the saved files from the first startup.**

### Install environment for the Dash app

```bash
cd IPD
conda create --name podcast_dash python=3.10 -y
conda activate podcast_dash
pip install -r requirements.txt
```

### To run the dash visualization

```bash
cd IPD
conda activate podcast_dash
export OPENAI_API_KEY="sk-YOURKEYHERE"
export GOOGLE_API_KEY="YOURKEYHERE"
export PYTHONPATH="$PYTHONPATH:$PWD"
python src/main.py 
```

---

## Podcast Transcription Pipeline

The transcription pipeline automatically downloads and transcribes podcast episodes using WhisperX (Whisper large-v3 + voice activity detection). It supports processing individual audio files, entire RSS feeds, or batch processing multiple podcasts from a JSON configuration file.

### Requirements

**Hardware:**

- GPU recommended (CUDA or MPS) for faster processing
- CPU fallback available for transcription

**Models:**

- WhisperX large-v3 (multilingual, 1550M parameters)
- Language-specific alignment models (automatically downloaded)

### Environment for transcription

The podcast_dash environment can be used for transcription.

### Usage examples

All files are expected to be run from the project root directory.

**Process a single audio file:**

```bash
python IPD/src/data_pipeline/transcription/file2transcript.py /path/to/audio/file.mp3
```

**Process an entire podcast RSS feed:**

```bash
python IPD/src/data_pipeline/transcription/file2transcript.py "https://feeds.simplecast.com/your-podcast-feed"
```

**Batch process multiple podcasts from JSON:**

```bash
python IPD/src/data_pipeline/transcription/file2transcript.py --input IPD/src/data_pipeline/transcription/podcast_feeds.json
```

### Output

Transcripts are saved in structured JSON format with:

- Word-level timestamps and confidence scores
- Episode metadata (title, description, publication date)
- Segmented transcriptions for downstream processing

Files are organized in `IPD/src/data_pipeline/transcription/raw_data/<podcast_name>/` with automatic progress tracking for interrupted batch jobs.

**Generated datasets:**

```bash
python IPD/src/data_pipeline/transcription/items2dataset.py  # Converts raw JSON to Parquet/CSV datasets
```

Files are organized in `dataset/parquet/<podcast_name>.parquet` and/or `dataset/csv/<podcast_name>.csv`

---

## Transcript Segmentation Pipeline

To run this code a Google API key is required.

```bash
export GOOGLE_API_KEY="YOURKEYHERE"
```

All in this part of the pipeline can be run as Jupyter notebooks files:

- Create segmentations src/datapipeline/segmenting/create_segments.ipynb
- Highlight segmentations src/datapipeline/segmenting/highlight_segments.ipynb
- Character timeline src/datapipeline/segmenting/create_segments.ipynb

## Authors

- @anetey-abbey
- @pascalmathas
- @robbertjs
- @Joppewouts

### Acknowledgements

Thanks to [Whispering-GPT](https://huggingface.co/Whispering-GPT) for transcribing the [Lex Fridman Podcast](https://huggingface.co/datasets/Whispering-GPT/lex-fridman-podcast), which is used in this project.

This work was carried out on the Dutch national e-infrastructure with the support of [SURF Cooperative](http://surfsara.nl/)
