"""Smoke test: does the z.ai API accept an audio file and return a transcript?

Usage:
    export ZAI_API_KEY=your-key
    python test_audio.py path/to/clip.mp3            # normal request
    python test_audio.py path/to/clip.wav --stream   # streamed transcript
    python test_audio.py path/to/video.mp4           # mp4, any length

Limits for glm-asr-2512 (per z.ai docs): .wav or .mp3, <= 25 MB, <= 30 seconds.
The API does not take mp4, so an mp4 is converted with ffmpeg and cut into 25 s
WAV chunks. The chunks are uploaded one after another, each with the text so
far sent as `prompt` context. Needs ffmpeg on PATH (or `pip install imageio-ffmpeg`).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import requests

BASE_URL = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4")
MODEL = "glm-asr-2512"
MAX_BYTES = 25 * 1024 * 1024
MIME_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav"}
CHUNK_SECONDS = 25
# z.ai recommends keeping the context prompt under 8000 characters.
MAX_CONTEXT_CHARS = 8000


def find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("mp4 needs ffmpeg: install it or `pip install imageio-ffmpeg`.")


def split_mp4(src, out_dir):
    """Extract the audio track of `src` as 16 kHz mono WAV chunks of CHUNK_SECONDS."""
    cmd = [
        find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
        os.path.join(out_dir, "chunk_%04d.wav"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{result.stderr.strip()}")
    return sorted(glob.glob(os.path.join(out_dir, "chunk_*.wav")))


def transcribe_mp4(api_key, src):
    with tempfile.TemporaryDirectory() as tmp:
        chunks = split_mp4(src, tmp)
        text = ""
        for i, chunk in enumerate(chunks, 1):
            data = {"model": MODEL, "stream": "false"}
            if text:
                data["prompt"] = text[-MAX_CONTEXT_CHARS:]
            with open(chunk, "rb") as f:
                resp = requests.post(
                    f"{BASE_URL}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                    files={"file": (os.path.basename(chunk), f, "audio/wav")},
                    timeout=120,
                )
            print(f"chunk {i}/{len(chunks)}: HTTP {resp.status_code}")
            if not resp.ok:
                print(resp.text)
                sys.exit(1)
            text = f"{text} {resp.json().get('text', '').strip()}".strip()
        print("\nTranscript:\n" + text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="path to a .wav or .mp3 file (30 s max), or an .mp4")
    parser.add_argument("--stream", action="store_true", help="request a streamed transcript")
    args = parser.parse_args()

    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        sys.exit("Set ZAI_API_KEY first.")

    ext = os.path.splitext(args.audio)[1].lower()
    if ext == ".mp4":
        transcribe_mp4(api_key, args.audio)
        return
    if ext not in MIME_TYPES:
        sys.exit(f"Unsupported extension {ext!r}; use .wav, .mp3 or .mp4.")
    size = os.path.getsize(args.audio)
    if size > MAX_BYTES:
        sys.exit(f"File is {size / 1e6:.1f} MB; the limit is 25 MB.")

    with open(args.audio, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": MODEL, "stream": str(args.stream).lower()},
            files={"file": (os.path.basename(args.audio), f, MIME_TYPES[ext])},
            stream=args.stream,
            timeout=120,
        )

    print(f"HTTP {resp.status_code}")
    if not resp.ok:
        print(resp.text)
        sys.exit(1)

    if args.stream:
        # Server-sent events: print each data payload as it arrives.
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                print(payload)
    else:
        body = resp.json()
        print(json.dumps(body, indent=2, ensure_ascii=False))
        print("\nTranscript:\n" + body.get("text", "<no 'text' field in response>"))


if __name__ == "__main__":
    main()
