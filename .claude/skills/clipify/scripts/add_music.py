#!/usr/bin/env python3
"""Mix a background music track under a rendered clip.

The music is trimmed (or looped) to the clip's length, faded in/out, and mixed
under the clip's own audio. With --duck, the music dips automatically whenever
someone is talking (sidechain compression keyed on the clip audio), so dialogue
stays intelligible. Video is stream-copied — no re-encode.

Usage:
    python3 add_music.py clip.mp4 music.mp3 out.mp4 \
        [--music-start 12.5] [--music-vol 0.25] [--original-vol 1.0] \
        [--duck] [--mute-original] [--fade 0.8]

Tuning notes:
- music-start skips into the track (e.g. straight to the hook/drop).
- music-vol 0.25 sits under dialogue; 0.4–0.6 for action montages where the
  strike sounds should punch through but the beat carries the edit.
- --duck for dialogue clips. Skip it for action mode — ducking on every strike
  makes the music pump.
- --mute-original replaces the clip audio entirely (music-only edit).
- Clips without an audio stream are treated as --mute-original.
"""
import sys, json, argparse, subprocess


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    duration = float(info["format"]["duration"])
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return duration, has_audio


def build_filter(dur, music_vol, original_vol, fade, duck, keep_original):
    fade = min(fade, dur / 2)
    music = (f"[1:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,volume={music_vol},"
             f"afade=t=in:d={fade:.3f},afade=t=out:st={dur - fade:.3f}:d={fade:.3f}")
    if not keep_original:
        return f"{music},alimiter=limit=0.95[a]"
    if duck:
        return (f"{music}[m];"
                f"[0:a]asplit=2[o][sc];"
                f"[m][sc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[md];"
                f"[o]volume={original_vol}[ov];"
                f"[ov][md]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
    return (f"{music}[m];"
            f"[0:a]volume={original_vol}[ov];"
            f"[ov][m]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("music")
    ap.add_argument("out")
    ap.add_argument("--music-start", type=float, default=0.0)
    ap.add_argument("--music-vol", type=float, default=0.25)
    ap.add_argument("--original-vol", type=float, default=1.0)
    ap.add_argument("--fade", type=float, default=0.8)
    ap.add_argument("--duck", action="store_true")
    ap.add_argument("--mute-original", action="store_true")
    a = ap.parse_args()

    dur, has_audio = probe(a.clip)
    keep_original = has_audio and not a.mute_original
    graph = build_filter(dur, a.music_vol, a.original_vol, a.fade, a.duck, keep_original)

    cmd = ["ffmpeg", "-y", "-i", a.clip,
           "-stream_loop", "-1", "-ss", str(a.music_start), "-i", a.music,
           "-filter_complex", graph,
           "-map", "0:v", "-map", "[a]",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-t", f"{dur:.3f}", "-movflags", "+faststart", a.out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
        sys.exit(r.returncode)
    print(a.out)


if __name__ == "__main__":
    main()
