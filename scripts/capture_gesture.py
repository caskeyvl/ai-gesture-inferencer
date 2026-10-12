#!/usr/bin/env python3
"""Capture labeled STM32F4 Discovery accelerometer gestures over UART.

This script is designed for the ColoradoEmbeddedAI
stm32f4_accelerometer_example firmware. The board waits for a command of
this form:

    START <n>

and then streams n lines like:

    X=-988  Y=32  Z=-16 mg

Each capture is saved directly into this Lab 2 layout:

    data/raw/<gesture>/p##_s##_<gesture>_###.csv

CSV schema:

    timestamp,accel_x,accel_y,accel_z

`timestamp` is elapsed host time in seconds from receipt of the first sample.
Accelerometer values are stored in milligravity (mg), matching the firmware.

Example (Windows PowerShell):
    python scripts/capture_gesture.py --port COM5 --gesture circle \
        --participant 1 --session 1 --reps 10 --samples 100

Example (Linux / WSL with USB-UART exposed):
    python3 scripts/capture_gesture.py --port /dev/ttyUSB0 --gesture circle \
        --participant 1 --session 1 --reps 10 --samples 100
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

try:
    import serial
except ImportError:
    sys.exit(
        "pyserial is required. Install it in your venv with:\n"
        "  pip install pyserial"
    )


BAUD_RATE = 115200
SAMPLE_LINE_RE = re.compile(
    r"^X=(-?\d+)\s+Y=(-?\d+)\s+Z=(-?\d+)\s+mg$"
)
ACK_LINE_RE = re.compile(r"^START ACK (\d+) chip=(\S+)$")
GESTURE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture labeled gesture recordings from STM32F4 Discovery UART."
    )
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port, e.g. COM5 on Windows or /dev/ttyUSB0 on Linux.",
    )
    parser.add_argument(
        "--gesture",
        required=True,
        help="Gesture/class name, e.g. circle, shake, up_down, idle.",
    )
    parser.add_argument(
        "--participant",
        required=True,
        help="Participant number or ID, e.g. 1 or p01.",
    )
    parser.add_argument(
        "--session",
        required=True,
        help="Session number or ID, e.g. 1 or s01.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Number of repetitions to capture in this run (default: 1).",
    )
    parser.add_argument(
        "--start-rep",
        type=int,
        default=1,
        help="First repetition number used in filenames (default: 1).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Samples requested from the board per repetition (default: 100).",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Countdown seconds before each capture (default: 3).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory for raw gesture data (default: data/raw).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing capture file.",
    )
    return parser.parse_args()


def normalize_index(value: str, prefix: str) -> str:
    value = value.strip().lower()
    if value.startswith(prefix):
        value = value[len(prefix) :]
    if not value.isdigit():
        raise ValueError(f"Expected {prefix} followed by a number, got {value!r}.")
    number = int(value)
    if number < 0 or number > 99:
        raise ValueError(f"{prefix} number must be between 0 and 99.")
    return f"{prefix}{number:02d}"


def validate_args(args: argparse.Namespace) -> tuple[str, str, str]:
    gesture = args.gesture.strip().lower().replace(" ", "_")
    if not GESTURE_RE.fullmatch(gesture):
        raise ValueError(
            "Gesture names may contain only letters, numbers, '_' and '-', "
            "and must start with a letter or number."
        )
    if args.reps < 1:
        raise ValueError("--reps must be at least 1.")
    if args.start_rep < 1 or args.start_rep > 999:
        raise ValueError("--start-rep must be between 1 and 999.")
    if args.start_rep + args.reps - 1 > 999:
        raise ValueError("Requested repetitions would exceed repetition 999.")
    if args.samples < 1:
        raise ValueError("--samples must be at least 1.")
    if args.countdown < 0:
        raise ValueError("--countdown must be 0 or greater.")

    participant = normalize_index(str(args.participant), "p")
    session = normalize_index(str(args.session), "s")
    return gesture, participant, session


def read_line(ser: serial.Serial) -> str:
    raw = ser.readline()
    if not raw:
        raise TimeoutError("Serial read timed out.")
    return raw.decode("ascii", errors="replace").strip()


def send_start(ser: serial.Serial, samples: int) -> None:
    command = f"START {samples}\r\n".encode("ascii")
    ser.write(command)
    ser.flush()


def wait_for_ack(ser: serial.Serial, samples: int) -> str:
    while True:
        line = read_line(ser)
        if line:
            print(f"BOARD: {line}")
        match = ACK_LINE_RE.fullmatch(line)
        if match and int(match.group(1)) == samples:
            return match.group(2)


def capture_samples(
    ser: serial.Serial,
    samples: int,
    writer: csv.writer,
) -> None:
    samples_seen = 0
    first_sample_time: float | None = None

    while samples_seen < samples:
        line = read_line(ser)
        match = SAMPLE_LINE_RE.fullmatch(line)
        if not match:
            if line:
                print(f"BOARD: {line}")
            continue

        now = time.monotonic()
        if first_sample_time is None:
            first_sample_time = now

        timestamp = now - first_sample_time
        x, y, z = match.groups()
        writer.writerow([f"{timestamp:.6f}", x, y, z])
        samples_seen += 1

        print(
            f"\rSamples: {samples_seen:4d}/{samples}   "
            f"X={x:>6} Y={y:>6} Z={z:>6} mg",
            end="",
            flush=True,
        )

    print()


def wait_for_done(ser: serial.Serial) -> None:
    """Consume firmware output until DONE so the next capture starts cleanly."""
    while True:
        line = read_line(ser)
        if line:
            print(f"BOARD: {line}")
        if line == "DONE":
            return


def countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"Starting in {remaining}...", flush=True)
        time.sleep(1)
    print("GO!", flush=True)


def capture_one(
    ser: serial.Serial,
    out_path: Path,
    samples: int,
    countdown_seconds: int,
    overwrite: bool,
) -> str:
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Capture already exists: {out_path}\n"
            "Use a different repetition number or pass --overwrite intentionally."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    input(f"\nReady for {out_path.name}. Press Enter when the participant is ready...")
    countdown(countdown_seconds)

    # Drop stale banner/READY text. The capture's ACK contains the chip name,
    # so nothing important is lost here.
    ser.reset_input_buffer()
    send_start(ser, samples)
    chip = wait_for_ack(ser, samples)

    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "accel_x", "accel_y", "accel_z"])
            capture_samples(ser, samples, writer)
        wait_for_done(ser)
        temp_path.replace(out_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return chip


def main() -> int:
    args = parse_args()

    try:
        gesture, participant, session = validate_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_dir = args.out_root / gesture
    output_dir.mkdir(parents=True, exist_ok=True)

    print("STM32F4 Gesture Capture")
    print("=======================")
    print(f"Port        : {args.port}")
    print(f"Baud        : {BAUD_RATE}")
    print(f"Gesture     : {gesture}")
    print(f"Participant : {participant}")
    print(f"Session     : {session}")
    print(f"Repetitions : {args.reps}")
    print(f"Samples/rep : {args.samples}")
    print(f"Output      : {output_dir}")
    print("CSV units   : timestamp=seconds, acceleration=mg")

    try:
        with serial.Serial(args.port, BAUD_RATE, timeout=10) as ser:
            # Give the serial connection a moment to settle.
            time.sleep(0.25)

            for offset in range(args.reps):
                rep = args.start_rep + offset
                filename = (
                    f"{participant}_{session}_{gesture}_{rep:03d}.csv"
                )
                out_path = output_dir / filename

                chip = capture_one(
                    ser=ser,
                    out_path=out_path,
                    samples=args.samples,
                    countdown_seconds=args.countdown,
                    overwrite=args.overwrite,
                )

                print(f"Saved       : {out_path}")
                print(f"Sensor chip : {chip}")

    except serial.SerialException as exc:
        print(f"\nSERIAL ERROR: {exc}", file=sys.stderr)
        print(
            "Check the COM/tty port, USB-UART wiring, and that no other terminal "
            "program has the port open.",
            file=sys.stderr,
        )
        return 1
    except (TimeoutError, FileExistsError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCapture cancelled by user.")
        return 130

    print("\nAll requested captures completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
