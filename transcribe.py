"""Transcribe audio or video of any length with z.ai's glm-asr-2512.

glm-asr-2512 only takes .wav/.mp3 files up to 30 seconds and 25 MB. To get
around that, the input (mp4, m4a, mov, long mp3, ...) is converted with ffmpeg
to mono 16 kHz WAV and cut into 25-second chunks. The chunks are uploaded one
after another, and each request carries the transcript so far as `prompt`
context so the model can keep names, terms and sentences consistent across
chunk boundaries.

Usage:
    export ZAI_API_KEY=your-key
    python transcribe.py meeting.mp4
    python transcribe.py meeting.mp4 -o transcript.txt

Requires ffmpeg on PATH (or `pip install imageio-ffmpeg`). Without ffmpeg,
only .wav/.mp3 files of 30 seconds or less can be sent, as-is.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import requests

BASE_URL = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4")
MODEL = "glm-asr-2512"
CHUNK_SECONDS = 25
MAX_BYTES = 25 * 1024 * 1024
# z.ai recommends keeping the context prompt under 8000 characters.
MAX_CONTEXT_CHARS = 8000
MIME_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav"}


def find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def split_audio(ffmpeg, src, out_dir, chunk_seconds=CHUNK_SECONDS):
    """Extract the audio track of `src` and cut it into WAV chunks of `chunk_seconds`."""
    pattern = os.path.join(out_dir, "chunk_%04d.wav")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", src,
        "-vn",                   # drop any video stream
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "segment", "-segment_time", str(chunk_seconds),
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.strip()}")
    chunks = sorted(glob.glob(os.path.join(out_dir, "chunk_*.wav")))
    if not chunks:
        raise RuntimeError(f"ffmpeg produced no audio from {src} (does it have an audio track?)")
    return chunks


def transcribe_chunk(api_key, path, context=""):
    ext = os.path.splitext(path)[1].lower()
    data = {"model": MODEL, "stream": "false"}
    if context:
        data["prompt"] = context[-MAX_CONTEXT_CHARS:]
    with open(path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files={"file": (os.path.basename(path), f, MIME_TYPES[ext])},
            timeout=120,
        )
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code} for {os.path.basename(path)}: {resp.text}")
    return resp.json().get("text", "").strip()


def transcribe_file(api_key, src, on_chunk=None):
    """Transcribe `src` chunk by chunk; returns the full transcript.

    `on_chunk(index, total, text)` is called after each chunk is transcribed.
    """
    ffmpeg = find_ffmpeg()
    ext = os.path.splitext(src)[1].lower()

    if ffmpeg is None:
        if ext not in MIME_TYPES:
            raise RuntimeError(f"{ext or 'this'} files need ffmpeg to convert them; install ffmpeg "
                               "or `pip install imageio-ffmpeg`.")
        if os.path.getsize(src) > MAX_BYTES:
            raise RuntimeError("File is over 25 MB; install ffmpeg so it can be split.")
        print("ffmpeg not found; sending the file as-is (z.ai rejects anything over 30 s).",
              file=sys.stderr)
        text = transcribe_chunk(api_key, src)
        if on_chunk:
            on_chunk(1, 1, text)
        return text

    with tempfile.TemporaryDirectory(prefix="zai-chunks-") as tmp:
        chunks = split_audio(ffmpeg, src, tmp)
        transcript = ""
        for i, chunk in enumerate(chunks, 1):
            try:
                text = transcribe_chunk(api_key, chunk, context=transcript)
            except Exception as e:
                raise RuntimeError(f"Chunk {i}/{len(chunks)} failed: {e}\n\n"
                                   f"Transcript so far:\n{transcript}") from e
            if text:
                transcript = f"{transcript} {text}".strip()
            if on_chunk:
                on_chunk(i, len(chunks), text)
        return transcript


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="audio or video file (mp4, mp3, wav, m4a, ...)")
    parser.add_argument("-o", "--output", help="also write the full transcript to this file")
    args = parser.parse_args()

    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        sys.exit("Set ZAI_API_KEY first.")

    def progress(i, total, text):
        print(f"[{i}/{total}] {text}", file=sys.stderr, flush=True)

    try:
        transcript = transcribe_file(api_key, args.input, on_chunk=progress)
    except RuntimeError as e:
        sys.exit(str(e))

    print(transcript)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(transcript + "\n")


if __name__ == "__main__":
    main()
