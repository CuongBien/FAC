#!/usr/bin/env python3
"""PTDB-TUG Dataset Downloader.

Downloads microphone speech (.wav) and reference pitch ground truth (.f0)
from Graz University of Technology (SPSC Lab).
URL: http://www2.spsc.tugraz.at/databases/PTDB-TUG/

Reference:
    Pirker et al., 'A Pitch Tracking Corpus with Evaluation on Multipitch Tracking Scenario',
    Interspeech 2011.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import sys
import urllib.request
import urllib.error

BASE_URL = "http://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH%20DATA"


def get_available_sentences(gender_dir: str, speaker: str) -> list[str]:
    """Query PTDB-TUG HTTP directory listing to discover all available sentence IDs."""
    url = f"{BASE_URL}/{gender_dir}/MIC/{speaker}/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        matches = re.findall(rf"mic_{speaker}_([a-zA-Z0-9]+)\.wav", html)
        # Deduplicate while preserving order
        unique = []
        for m in matches:
            if m not in unique:
                unique.append(m)
        return sorted(unique)
    except Exception as e:
        print(f"[!] Warning: Could not fetch directory listing for {speaker} from {url}: {e}")
        # Fallback to standard TIMIT sentence IDs
        return [
            "sa1", "sa2", "sx10", "sx100", "sx109", "sx118", "sx127",
            "sx136", "sx145", "sx154", "sx163", "sx172", "sx181", "sx19"
        ]


def download_file(url: str, dest_path: str, timeout: int = 20) -> bool:
    """Download a single file if it does not already exist."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(temp_path, "wb") as f:
            while chunk := resp.read(65536):
                f.write(chunk)
        os.replace(temp_path, dest_path)
        return True
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"[!] Failed to download {url}: {e}")
        return False


def download_ptdb_subset(
    speakers: list[str],
    num_utterances: int,
    out_dir: str = "data/ptdb_tug",
    max_workers: int = 4,
):
    """Download PTDB-TUG audio and reference pitch files for given speakers."""
    print("=" * 70)
    print("PTDB-TUG Dataset Downloader (Graz University of Technology)")
    print("=" * 70)
    print(f"Target speakers   : {', '.join(speakers)}")
    print(f"Utterances/speaker: {num_utterances if num_utterances > 0 else 'ALL'}")
    print(f"Output directory  : {out_dir}")
    print(f"Concurrent workers: {max_workers}")
    print("-" * 70)

    download_tasks = []

    for spk in speakers:
        spk = spk.strip().upper()
        if spk.startswith("F"):
            gender_dir = "FEMALE"
        elif spk.startswith("M"):
            gender_dir = "MALE"
        else:
            print(f"[!] Unknown speaker format '{spk}'. Skipping.")
            continue

        print(f"[*] Discovering sentences for speaker {spk} ({gender_dir})...")
        sentences = get_available_sentences(gender_dir, spk)
        if num_utterances > 0:
            sentences = sentences[:num_utterances]
        print(f"    Found {len(sentences)} sentences to download for {spk}.")

        mic_dir = os.path.join(out_dir, gender_dir, "MIC", spk)
        ref_dir = os.path.join(out_dir, gender_dir, "REF", spk)

        for sent in sentences:
            # Microphone audio WAV
            wav_name = f"mic_{spk}_{sent}.wav"
            wav_url = f"{BASE_URL}/{gender_dir}/MIC/{spk}/{wav_name}"
            wav_path = os.path.join(mic_dir, wav_name)
            download_tasks.append((wav_url, wav_path, f"{spk} {sent} WAV"))

            # Reference ground-truth F0
            ref_name = f"ref_{spk}_{sent}.f0"
            ref_url = f"{BASE_URL}/{gender_dir}/REF/{spk}/{ref_name}"
            ref_path = os.path.join(ref_dir, ref_name)
            download_tasks.append((ref_url, ref_path, f"{spk} {sent} F0"))

    total_tasks = len(download_tasks)
    print(f"[*] Starting download of {total_tasks} files...")

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url, path): desc for (url, path, desc) in download_tasks}
        for i, future in enumerate(as_completed(futures), 1):
            desc = futures[future]
            res = future.result()
            if res:
                success_count += 1
            if i % 10 == 0 or i == total_tasks:
                sys.stdout.write(f"\rProgress: [{i}/{total_tasks}] (Success: {success_count})")
                sys.stdout.flush()

    print()
    print("-" * 70)
    print(f"[+] Download complete: {success_count}/{total_tasks} files downloaded.")
    print(f"[+] Data saved to: {os.path.abspath(out_dir)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Download subset of PTDB-TUG speech database.")
    parser.add_argument(
        "--speakers",
        type=str,
        default="F01,M01",
        help="Comma-separated speaker IDs (default: 'F01,M01'). Options: F01-F10, M01-M10.",
    )
    parser.add_argument(
        "--num-utterances",
        type=str,
        default="10",
        help="Number of sentences per speaker (default: 10, or 'all').",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/ptdb_tug",
        help="Target folder for dataset (default: 'data/ptdb_tug').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download threads (default: 4).",
    )
    args = parser.parse_args()

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    if args.num_utterances.lower() == "all":
        num_utterances = 0
    else:
        num_utterances = int(args.num_utterances)

    download_ptdb_subset(
        speakers=speakers,
        num_utterances=num_utterances,
        out_dir=args.out_dir,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
