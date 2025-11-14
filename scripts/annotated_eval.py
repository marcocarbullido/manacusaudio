import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import shlex
import subprocess
import tempfile
from typing import List, Dict, Any

import pandas as pd
from moviepy.config import FFMPEG_BINARY
from moviepy.video.io.VideoFileClip import VideoFileClip

from audio_processing.box_download import BoxNavigator
from scripts.predict import load_audio_model, predict_with_context, LABEL_MAP, AudioModelContext


def normalize_label(label: str | None) -> str | None:
    """Reduce labels to {'s', 'g', 'n'}; keep None when no annotation is provided."""
    if label is None:
        return None
    label = label.strip().lower()
    if not label:
        return None
    if label in {"s", "g"}:
        return label
    return "n"


def parse_annotation_cell(cell_value: Any) -> List[Dict[str, float]]:
    """Convert a cell like '0.5-1.0:s, 1.1-1.4:n' into structured windows."""
    if not isinstance(cell_value, str):
        return []

    windows = []
    for raw_chunk in cell_value.split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            window_part, raw_label = chunk.rsplit(":", 1)
        else:
            window_part, raw_label = chunk, None
        label = normalize_label(raw_label)
        if "-" not in window_part:
            continue
        start_str, end_str = window_part.split("-", 1)
        start = _safe_float(start_str)
        end = _safe_float(end_str)
        if start is None or end is None or not math.isfinite(start) or not math.isfinite(end):
            continue
        if end <= start:
            continue
        windows.append({"start": start, "end": end, "label": label})
    return windows


def _safe_float(value: str):
    try:
        return float(value.strip().replace(",", "."))
    except (ValueError, AttributeError):
        return None


def _select_allowed_label(probabilities, allowed_labels=None):
    """Return the highest-probability label restricted to allowed labels."""
    allowed_labels = allowed_labels or {"s", "g", "n"}
    best_label = None
    best_prob = float("-inf")
    for idx, label in LABEL_MAP.items():
        if label not in allowed_labels or idx >= len(probabilities):
            continue
        prob = float(probabilities[idx])
        if prob > best_prob:
            best_prob = prob
            best_label = label
    return best_label


def load_annotation_entries(spreadsheet_path: str) -> List[Dict[str, Any]]:
    df = pd.read_excel(spreadsheet_path)
    entries = []
    for _, row in df.iterrows():
        filename = row.get("filename")
        jump_windows = row.get("jump windows")
        if not isinstance(filename, str):
            continue
        windows = parse_annotation_cell(jump_windows)
        if not windows:
            continue
        entries.append({"filename": filename.strip(), "windows": windows})
    return entries


def select_entries(entries, filenames=None, count=None, run_all=False):
    if filenames:
        lookup = set(filenames)
        selected = [entry for entry in entries if entry["filename"] in lookup]
    elif count is not None:
        selected = entries[:count]
    elif run_all:
        selected = entries
    else:
        selected = entries[:1]
    return selected


def ensure_video_local(navigator: BoxNavigator, video_name: str) -> str:
    navigator.download_vid(video_name)
    return os.path.join(navigator.download_dir, video_name)


def evaluate_video(
    entry, navigator: BoxNavigator, model_ctx: AudioModelContext, per_video_dir: str | None = None
) -> List[Dict[str, Any]]:
    video_name = entry["filename"]
    video_path = ensure_video_local(navigator, video_name)
    if not os.path.exists(video_path):
        print(f"Skipping {video_name}: not found locally after download attempt.")
        return []

    results: List[Dict[str, Any]] = []
    try:
        with VideoFileClip(video_path) as video_clip:
            if video_clip.audio is None:
                print(f"No audio track detected for {video_name}; skipping.")
                return results

            for window in entry["windows"]:
                start = window["start"]
                end = window["end"]
                expected = window["label"]
                tmp_path = _extract_window_to_temp(video_path, video_clip, start, end)
                if not tmp_path:
                    print(f"Failed to extract audio window {start}-{end} for {video_name}.")
                    continue
                try:
                    prediction, probabilities = predict_with_context(
                        model_ctx, tmp_path, return_probabilities=True
                    )
                finally:
                    os.remove(tmp_path)

                prob_dict = {LABEL_MAP[idx]: float(prob) for idx, prob in enumerate(probabilities)}
                normalized_annotation = expected  # already normalized (or None)
                normalized_prediction = _select_allowed_label(probabilities)
                match = (
                    None
                    if (normalized_annotation is None or normalized_prediction is None)
                    else normalized_annotation == normalized_prediction
                )
                results.append(
                    {
                        "filename": video_name,
                        "window_start": start,
                        "window_end": end,
                        "normalized_annotation": normalized_annotation,
                        "normalized_prediction": normalized_prediction,
                        "match": match,
                        **{f"prob_{label}": prob_dict[label] for label in LABEL_MAP.values()},
                    }
                )
    except OSError as exc:
        print(f"MoviePy could not open {video_name}: {exc}")

    if results and per_video_dir:
        os.makedirs(per_video_dir, exist_ok=True)
        csv_name = f"eval_{os.path.splitext(os.path.basename(video_name))[0]}_results.csv"
        csv_path = os.path.join(per_video_dir, csv_name)
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"Wrote per-video results to {csv_path}")

    return results


def _extract_window_to_temp(video_path: str, video_clip, start: float, end: float) -> str:
    """Try extracting audio via MoviePy; fall back to ffmpeg when necessary."""
    tmp_path = _extract_with_moviepy(video_clip, start, end)
    if tmp_path:
        return tmp_path
    return _extract_with_ffmpeg(video_path, start, end)


def _extract_with_moviepy(video_clip, start: float, end: float) -> str:
    try:
        subclip = video_clip.subclipped(start, end)
        if subclip is None or subclip.audio is None:
            return ""
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        subclip.audio.write_audiofile(tmp_path, codec="pcm_s16le", logger=None)
        subclip.close()
        return tmp_path
    except Exception as exc:  # pylint: disable=broad-except
        print(f"MoviePy extraction error ({start}-{end}): {exc}")
        return ""


def _extract_with_ffmpeg(video_path: str, start: float, end: float) -> str:
    tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_audio_path = tmp_audio.name
    tmp_audio.close()

    ffmpeg_cmd = shlex.split(FFMPEG_BINARY) if isinstance(FFMPEG_BINARY, str) else list(FFMPEG_BINARY)
    cmd = ffmpeg_cmd + [
        "-y",
        "-ss",
        f"{start:.2f}",
        "-to",
        f"{end:.2f}",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        tmp_audio_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.getsize(tmp_audio_path) == 0:
            raise RuntimeError("Empty audio file produced.")
        return tmp_audio_path
    except Exception as exc:  # pylint: disable=broad-except
        print(f"FFmpeg extraction error ({start}-{end}): {exc}")
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        return ""


def summarize_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No results to summarize.")
        return
    total = len(results)
    scored = [item for item in results if item["match"] is not None]
    scored_count = len(scored)
    per_class_stats = {
        label: {"tp": 0, "fp": 0, "fn": 0, "total": 0} for label in ["s", "g", "n"]
    }
    for item in scored:
        true_label = item["normalized_annotation"]
        predicted_label = item["normalized_prediction"]
        if true_label not in per_class_stats:
            continue
        per_class_stats[true_label]["total"] += 1
        if predicted_label == true_label:
            per_class_stats[true_label]["tp"] += 1
        else:
            per_class_stats[true_label]["fn"] += 1
            if predicted_label in per_class_stats:
                per_class_stats[predicted_label]["fp"] += 1

    if scored_count:
        correct = sum(1 for item in scored if item["match"])
        accuracy = correct / scored_count
        print(
            f"Processed {total} annotated windows | Accuracy: {accuracy:.2%} "
            f"({correct}/{scored_count} labeled windows)"
        )
    else:
        print(f"Processed {total} annotated windows | Accuracy: N/A (no labeled windows)")

    for label_key, label_name in [("s", "snap"), ("g", "grunt"), ("n", "nothing")]:
        stats = per_class_stats[label_key]
        total = stats["total"]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        if total:
            acc = tp / total
            precision = tp / (tp + fp) if (tp + fp) else None
            recall = tp / (tp + fn) if (tp + fn) else None
            precision_str = f"{precision:.2%} (tp={tp}, fp={fp})" if precision is not None else "N/A"
            recall_str = f"{recall:.2%} (tp={tp}, fn={fn})" if recall is not None else "N/A"
            print(
                f"{label_name.capitalize()} metrics — Accuracy: {acc:.2%} ({tp}/{total}), "
                f"Precision: {precision_str}, Recall: {recall_str}"
            )
        else:
            print(f"{label_name.capitalize()} metrics — Accuracy: N/A, Precision: N/A, Recall: N/A")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate annotated windows against the audio model.")
    parser.add_argument("--spreadsheet", required=True, help="Path to the annotation spreadsheet (.xlsx).")
    parser.add_argument("--model", required=True, help="Path to the trained audio model file.")
    parser.add_argument(
        "--box-base",
        default=os.getcwd(),
        help="Directory where Box downloads and credentials should live (defaults to CWD).",
    )
    parser.add_argument(
        "--output",
        default="annotation_eval_results.csv",
        help="Where to write the aggregated CSV of predictions.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional folder for per-video CSVs (named eval_<filename>_results.csv).",
    )
    parser.add_argument(
        "--filename",
        action="append",
        dest="filenames",
        help="Specific filename to evaluate (may be supplied multiple times).",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", type=int, help="Number of videos (from the top of the sheet) to evaluate.")
    selection.add_argument("--all", action="store_true", help="Evaluate every video in the sheet.")
    return parser.parse_args()


def main():
    args = parse_args()
    entries = load_annotation_entries(args.spreadsheet)
    if not entries:
        print("No annotated entries found in the spreadsheet.")
        return

    selected_entries = select_entries(entries, filenames=args.filenames, count=args.count, run_all=args.all)
    if not selected_entries:
        print("No entries matched the selection criteria.")
        return

    navigator = BoxNavigator(base_dir=args.box_base)
    model_ctx = load_audio_model(args.model)

    all_results: List[Dict[str, Any]] = []
    for entry in selected_entries:
        print(f"Evaluating {entry['filename']} ({len(entry['windows'])} annotated windows)...")
        all_results.extend(evaluate_video(entry, navigator, model_ctx, args.output_dir))

    summarize_results(all_results)
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output, index=False)
        print(f"Wrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
