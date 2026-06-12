#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf


def denoise_file(
    input_path: Path,
    output_path: Path,
    *,
    stationary: bool = False,
    prop_decrease: float = 1.0,
) -> None:
    audio, sr = librosa.load(str(input_path), sr=None, mono=False)
    channels = audio[np.newaxis] if audio.ndim == 1 else audio

    reduced = np.array(
        [
            nr.reduce_noise(
                y=ch,
                sr=sr,
                stationary=stationary,
                prop_decrease=prop_decrease,
                n_jobs=1,
            )
            for ch in channels
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), reduced.T, sr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Denoise audio")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--stationary", action="store_true")
    parser.add_argument("--strength", type=float, default=1.0, metavar="0-1")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"not found: {args.input}", file=sys.stderr)
        return 1

    output = args.output or args.input.with_stem(f"{args.input.stem}_cleaned").with_suffix(".wav")
    denoise_file(args.input, output, stationary=args.stationary, prop_decrease=args.strength)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
