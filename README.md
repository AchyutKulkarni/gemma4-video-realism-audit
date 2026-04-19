# CineRank AI

CineRank AI is a Gemma 4 multimodal demo for evaluating how realistic AI-generated videos look.

The competition-ready artifact in this repository is [notebooks/gemma4_transformers_demo.ipynb](notebooks/gemma4_transformers_demo.ipynb). It compares two AI-generated videos, samples representative frames, asks Gemma 4 to identify visible synthesis artifacts, aggregates multiple judge passes with consensus voting, and ranks the videos by realism.

## Competition Fit

For the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon), the strongest framing for this project is **digital trust / media authenticity**:
- both inputs are AI-generated videos
- Gemma 4 inspects visible artifacts such as face warping, identity drift, and texture flicker
- the system estimates which output is more visually convincing and therefore more risky in misinformation-sensitive settings

This keeps the demo tied to a real-world trust and resilience problem instead of a generic ranking demo.

## Main Demo

Use this notebook:
- [notebooks/gemma4_transformers_demo.ipynb](notebooks/gemma4_transformers_demo.ipynb)

Demo assets included in the repo:
- [demo_videos/good_video.mp4](demo_videos/good_video.mp4)
- [demo_videos/bad_video.mp4](demo_videos/bad_video.mp4)

## What The Notebook Does

1. Loads two AI-generated videos.
2. Extracts a small set of chronological frames from each video.
3. Calls `google/gemma-4-E2B-it` as a multimodal artifact judge.
4. Repeats artifact extraction multiple times.
5. Aggregates artifact labels by consensus vote.
6. Converts the consensus artifact report into deterministic realism scores.
7. Ranks the videos and prints a concise summary.

Both videos are treated as `ai_generated`. The notebook’s job is not provenance detection. Its job is to estimate **how realistic the synthetic result looks**.

## Output Schema

Each ranked result includes:
- `origin_verdict`
- `realism_label`
- `realism_score`
- `confidence`
- `artifact_evidence`
- `artifact_report`
- `vote_breakdown`

Interpretation:
- `ai_generated + high_realism` means the video is synthetic but visually convincing
- `ai_generated + low_realism` means visible artifacts make the video obviously synthetic

## Consensus Voting

Single Gemma runs were unstable across trials, so the notebook uses consensus voting:
- `CONSENSUS_RUNS = 3`
- `CONSENSUS_MAX_ATTEMPTS = 5`

Malformed model responses are skipped. The final result is computed from the successful runs only.

## Repo Layout

```text
cine-rank-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── demo_videos/
│   ├── good_video.mp4
│   └── bad_video.mp4
└── notebooks/
    └── gemma4_transformers_demo.ipynb
```

## Setup

For local work:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In practice, the easiest run path is a GPU-backed Colab notebook.

## Run Flow

1. Open `notebooks/gemma4_transformers_demo.ipynb` in Colab.
2. Enable a GPU runtime.
3. Set `VIDEO_PATHS` to your own `.mp4` files.
4. Provide `HF_TOKEN` via environment variable or enter it when prompted by the notebook.
5. Run the model-load cell.
6. Run the inference cell.
7. Review the summary and consensus vote breakdown.

Default model:

```python
MODEL_ID = "google/gemma-4-E2B-it"
```

## Requirements Notes

The notebook depends on:
- `torch`
- `transformers`
- `accelerate`
- `bitsandbytes`
- `opencv-python`
- `numpy`

Colab already provides some of these, but `requirements.txt` lists the core packages for clarity.

## Known Limitations

- The current demo compares only two videos.
- Gemma artifact labeling is still somewhat noisy, which is why consensus voting is required.
- This is a realism / artifact-auditing demo, not a full provenance-forensics system.
- The model can still overcall artifacts on better clips, so the output should be presented as a multimodal audit signal, not absolute ground truth.

## Suggested GitHub Description

> Gemma 4 multimodal demo for auditing realism in AI-generated videos using consensus artifact detection.
