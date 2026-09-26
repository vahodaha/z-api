"""Smoke test: does the z.ai API accept an audio file and return a transcript?

Usage:
    export ZAI_API_KEY=your-key
    python test_audio.py path/to/clip.mp3            # normal request
    python test_audio.py path/to/clip.wav --stream   # streamed transcript

Limits for glm-asr-2512 (per z.ai docs): .wav or .mp3, <= 25 MB, <= 30 seconds.
"""

import argparse
import json
import os
import sys

import requests

BASE_URL = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4")
MODEL = "glm-asr-2512"
MAX_BYTES = 25 * 1024 * 1024
MIME_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="path to a .wav or .mp3 file (30 s max)")
    parser.add_argument("--stream", action="store_true", help="request a streamed transcript")
    args = parser.parse_args()

    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        sys.exit("Set ZAI_API_KEY first.")

    ext = os.path.splitext(args.audio)[1].lower()
    if ext not in MIME_TYPES:
        sys.exit(f"Unsupported extension {ext!r}; glm-asr-2512 takes .wav or .mp3.")
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
