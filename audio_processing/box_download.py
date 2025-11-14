"""
Utility helpers for pulling annotated videos from Box and preparing local audio clips.

This module adapts the original Colab-only BoxNavigator notebook cell into a reusable
Python component that works anywhere (Colab, local dev machine, etc.). Paths default
to the current working directory but can be overridden when instantiating BoxNavigator.
"""

from __future__ import annotations

import os
import time
import shutil
import requests
from typing import Optional, Dict

from moviepy.video.io.VideoFileClip import VideoFileClip

try:  # Optional; available in notebooks but not required elsewhere
    from IPython.display import Audio, display, clear_output
except ImportError:  # pragma: no cover - fallback when IPython is missing
    Audio = None

    def display(*args, **kwargs):
        for arg in args:
            print(arg)

    def clear_output(*args, **kwargs):
        pass


class BoxNavigator:
    """
    Lightweight Box client powered by the REST API + Developer Token.

    - Credentials live under ``<base_dir>/system_files/box_credentials.txt``
    - Files download into ``<base_dir>/Downloaded_Videos``
    - Provide ``base_dir`` to control where downloads/creds are stored
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.home_dir = os.path.abspath(base_dir or os.getcwd())
        self.system_files_dir = os.path.join(self.home_dir, "system_files")
        self.download_dir = os.path.join(self.home_dir, "Downloaded_Videos")

        os.makedirs(self.home_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)

        if os.path.exists(self._cred_file_path()):
            self.load_credentials_from_file()
        else:
            self.setup_credentials()

        self._build_session()

    # ---------- Credentials ----------
    def _cred_file_path(self) -> str:
        return os.path.join(self.system_files_dir, "box_credentials.txt")

    def load_credentials_from_file(self) -> None:
        os.makedirs(self.system_files_dir, exist_ok=True)
        with open(self._cred_file_path(), "r", encoding="utf-8") as file:
            lines = [ln.strip() for ln in file.readlines()]

        self.client_id = lines[0] if len(lines) >= 1 else ""
        self.client_secret = lines[1] if len(lines) >= 2 else ""
        self.access_token = lines[2] if len(lines) >= 3 else ""
        if not self.access_token:
            self.access_token = lines[0] if lines else ""

        if not self.access_token:
            self.access_token = input("Enter your Box Developer Token: ").strip()
            with open(self._cred_file_path(), "w", encoding="utf-8") as f:
                f.write(f"{self.client_id}\n{self.client_secret}\n{self.access_token}")

    def setup_credentials(self) -> None:
        os.makedirs(self.system_files_dir, exist_ok=True)
        print("Login to https://tulane.app.box.com/developers/console and create a Developer Token.")
        self.client_id = input("Enter your Box Client ID (optional, press Enter to skip): ").strip()
        self.client_secret = input("Enter your Box Client Secret (optional, press Enter to skip): ").strip()
        self.access_token = input("Enter your Box Developer Token: ").strip()

        with open(self._cred_file_path(), "w", encoding="utf-8") as file:
            file.write(f"{self.client_id}\n{self.client_secret}\n{self.access_token}")

    def _build_session(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.access_token}"})
        self.base_url = "https://api.box.com/2.0"

    def _update_token(self) -> None:
        self.access_token = input("Your Box Developer Token seems expired. Enter a new token: ").strip()
        with open(self._cred_file_path(), "w", encoding="utf-8") as f:
            f.write(f"{self.client_id}\n{self.client_secret}\n{self.access_token}")
        self._build_session()

    def _request(self, method: str, url: str, **kwargs):
        resp = self.session.request(method, url, **kwargs)
        if resp.status_code == 401:
            self._update_token()
            resp = self.session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp

    # ---------- Box operations ----------
    def search(self, video_name: str) -> Optional[Dict[str, str]]:
        params = {
            "query": video_name,
            "type": "file",
            "content_types": "name",
            "limit": 25,
        }
        url = f"{self.base_url}/search"
        try:
            data = self._request("GET", url, params=params).json()
        except requests.HTTPError as exc:
            print(f"Search error: {exc}")
            return None

        entries = data.get("entries", [])
        for item in entries:
            if item.get("type") == "file" and item.get("name") == video_name:
                return {"id": item.get("id"), "name": item.get("name"), "type": "file"}
        return None

    def download_vid(self, video_name: str) -> None:
        video_path = os.path.join(self.download_dir, video_name)
        if os.path.exists(video_path):
            print(f"Video '{video_name}' already exists at: {video_path}")
            return

        result = self.search(video_name)
        if result and result["type"] == "file" and video_name.lower().endswith(".mp4"):
            file_id = result["id"]
            content_url = f"{self.base_url}/files/{file_id}/content"
            print(f"Downloading '{video_name}' to: {video_path} ...")
            try:
                with self._request("GET", content_url, stream=True) as response:
                    with open(video_path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                print(f"Successfully saved '{video_name}'.")
            except requests.HTTPError as exc:
                print(f"Download failed: {exc}")
        else:
            print(f"Video '{video_name}' not found in Box by exact name, or is not an mp4 file.")

    # ---------- Local audio ops ----------
    def extract_audio_clips(self, video_name: str) -> Optional[str]:
        video_path = os.path.join(self.download_dir, video_name)
        valid_ext = video_name.lower().endswith((".mp4", ".avi"))
        if not (os.path.isfile(video_path) and valid_ext):
            print(f"Video file '{video_name}' not found or not a supported type (.mp4/.avi).")
            return None

        try:
            video_name_without_ext = os.path.splitext(video_name)[0]
            audio_clips_folder_path = os.path.join(self.download_dir, video_name_without_ext)
            os.makedirs(audio_clips_folder_path, exist_ok=True)

            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            if audio_clip is None:
                print("No audio track detected.")
                video_clip.close()
                return None

            duration = float(audio_clip.duration or 0.0)
            i = 0.0
            while i < duration:
                start_time = i
                end_time = min(i + 0.5, duration)
                subclip = audio_clip.subclipped(start_time, end_time)

                clip_filename = f"{video_name_without_ext}_{int(round(i * 2)):04d}.wav"
                clip_filepath = os.path.join(audio_clips_folder_path, clip_filename)

                if not os.path.exists(clip_filepath):
                    subclip.write_audiofile(clip_filepath, codec="pcm_s16le", logger=None)
                    print(f"Saved clip: {clip_filename}")

                i += 0.5

            try:
                video_clip.reader.close()
            except Exception:
                pass
            try:
                if hasattr(audio_clip, "reader"):
                    audio_clip.reader.close_proc()
            except Exception:
                pass

            return audio_clips_folder_path

        except IOError as exc:
            print(f"An error occurred for {video_name}: {exc}")
            return None

    def labelclips(self, video_name: str) -> None:
        labels_dir = os.path.join(self.home_dir, "labels")
        labels = ["s", "g", "c", "c2", "r", "w", "n", "q"]
        for label in labels:
            os.makedirs(os.path.join(labels_dir, label), exist_ok=True)

        audio_clips_dir = os.path.join(self.download_dir, os.path.splitext(video_name)[0])
        if not os.path.exists(audio_clips_dir):
            print(f"There are no audio clips extracted for the video {video_name}.")
            return

        audio_clips = sorted(
            clip for clip in os.listdir(audio_clips_dir) if clip.lower().endswith(".wav")
        )
        n_clips_total = len(audio_clips)

        try:
            for idx, audio_clip_name in enumerate(audio_clips, start=1):
                clear_output(wait=True)
                remaining = n_clips_total - idx
                clip_path = os.path.join(audio_clips_dir, audio_clip_name)

                print(f"Playing {audio_clip_name}")
                if Audio is not None:
                    display(Audio(clip_path, autoplay=True))
                else:
                    print(f"(Audio playback unavailable. Inspect manually: {clip_path})")
                print(f"{remaining} clips remaining.")
                label = ""
                while label not in labels:
                    time.sleep(0.5)
                    label = input("Enter label (s, g, c, c2, r, w, n, q):\n").lower().strip()

                destination_path = os.path.join(labels_dir, label, audio_clip_name)
                shutil.move(clip_path, destination_path)
                print(f"Moved {audio_clip_name} to {destination_path}")
        except KeyboardInterrupt:
            print("Labeling interrupted by user.")


__all__ = ["BoxNavigator"]
