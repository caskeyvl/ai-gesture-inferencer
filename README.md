# Lab 2 - Accelerometer Gesture Dataset

## Project Overview

This project contains a custom accelerometer gesture dataset collected for Lab 2 of Embedded AI.

The goal of the project is to:

- Collect a representative 3-axis accelerometer dataset for several gesture classes.
- Organize and document the dataset in a consistent format.
- Apply meaningful data augmentation techniques.
- Demonstrate that the dataset can be used to train and evaluate a machine learning model.

---

# Team

| Team Member | Primary Role |
|---|---|
| Tyler Pilat | Project Infrastructure / validate_dataset file |
| Logan Caskey | Augment file / Quality Assurance |
| Joel Ramos Silva |  dataset / readme writeup|
| Julian Werder | Model Training / train_demo file |
| Jacob Thiel | Documentation / Final Integration |

All team members contributed to raw gesture data collection.

---

# Dataset

## Gesture Classes

The dataset contains the following gesture classes:

| Class | Description |
|---|---|
| `circle` | One clockwise loop traced in the vertical plane in front of the body, ~30 cm across (size of A4 sheet of paper).  |
| `square` | One clockwise square traced in the vertical plane, ~30 cm per side, with a brief pause at each corner.  |
| `triangle` | One clockwise equilateral triangle traced in the vertical plane, ~30 cm per side, with a brief pause at each corner. |
| `wave` | Repeated side-to-side horizontal sweeps, ~20 cm of travel, 4–6 passes. |
| `idle` | No gesture. Device held still and level, with normal resting hand tremor. |

## Gesture Definitions

### `circle`

All gestures are performed with the board held flat in the dominant hand,
components facing up, USB connector pointing toward the wrist.

Include:

- Starting position: Board flat and level, held at chest/neck height at the 12 o'clock point of the circle.
- Direction of motion: Clockwise, seen by the team member.
- Approximate motion size: About 30 cm in diameter
- End position: Back at the 12 o'clock starting point, one full loop completed, device held still.
- Any important restrictions:One loop only, completed in roughly one to two seconds. Smooth continuous speed — no stopping partway around. The board stays flat; do not rotate the wrist to follow the curve.

### `square`

Include:

- Starting position: Board flat and level at chest/neck height, at the top-left corner of the square
- Direction of motion: Clockwise — top-left edge, top right edge, bottom right edge, bottom left edge
- Approximate motion size: About 30 cm per side
- End position: Back at the top-left corner, device held still.
- Any important restrictions: One lap only, about two seconds total. Pause very briefly (roughly a tenth of a second) at each corner and move in straight lines between them. The pauses are
  what distinguish a square from a circle in the data, so do not round them off.

### `triangle`

- Starting position: Board flat and level at chest/neck height, at the bottom-left corner of the triangle.
- Direction of motion: Clockwise — up and right to the apex, down and right to the bottom-right corner, then straight left along the base.
- Approximate motion size: Roughly equilateral, about 30 cm per side.
- End position: Back at the bottom-left corner, device held still.
- Any important restrictions: One lap only, about 1.5 seconds. Pause briefly at each of the three corners, same as `square`. Straight edges between corners.


### `wave`

- Starting position: Board flat and level at chest height, arm centered in front of the body.
- Direction of motion: Side to side horizontally — right, then left, repeating. No vertical component.
- Approximate motion size: About 30 cm of travel in each direction.
- End position: Arm back near the center, device held still.
- Any important restrictions: 4–6 passes total within about two seconds, at a steady rhythm. Keep the board flat and level throughout, it's an arm motion not a wrist flick or a tilt.


### `idle`

- Starting position: Board flat and level, held comfortably at chest height.
- Direction of motion: None. Holds the device as still as you reasonably can.
- Approximate motion size: None intended. Only the small involuntary motion of a held hand.
- End position: Unchanged from the start.
- Any important restrictions: Hold for about two seconds. The participant should not deliberately stiffen or brace against a surface — the idea is to see what a hand at rest actually looks like to the accelerometer, including gravity on the stationary axes.

---

# Hardware

## Sensor / Development Board

Data was collected using:

- **Board:** STM32F4 Discovery (STM32F407VG), board revision MB997C/MB997D
- **Accelerometer:** STMicroelectronics LIS3DSH, 3-axis MEMS, 16-bit output, ±2 g full scale, read over SPI1
- **Accelerometer axes:** X, Y, Z
- **Sampling rate:** 50 Hz as recorded (see note below)
- **Acceleration units:** mg (milligravity; 1000 mg = 1 g)
- **Other relevant hardware:** USB-to-serial (USB-UART) adapter connected to USART2 on PA2/PA3 at 115200 8N1; ST-LINK/V2 (on-board) used only for flashing

## Sensor Placement

The sensor was:

The board was held flat in the participant's dominant hand, components facing up, with the USB connector pointing toward the wrist.
Participants were asked to hold the board level and to keep it level throughout each gesture — the traced shape is produced by moving the whole hand and
forearm, no tilting or rotating the wrist. Because the board is held level, gravity showed almost entirely on the Z axis at rest, which is visible in the data as Z ≈ 1000 mg with X and Y near zero.

Example:

> The development board was held in the participant's dominant hand with the USB connector facing toward the wrist. The elbow was bent roughly 90 degrees with the board at chest to neck height. Participants stood still; the gesture was not produced by walking or turning the body.

---

# Data Capture Software

Data was captured using:

- **Capture method:** STM32 UART capture over a USB-to-serial adapter
- **Software/tool:** `scripts/capture_gesture.py` (in this repository), paired with the course `stm32f4_accelerometer_example` firmware
- **Software version:** Python 3 with `pyserial`; firmware built from the course example repository with the print interval changed to 20 ms as described under Hardware
- **Computer/phone platform:** macOS (the script is platform-independent; only the serial port name differs — `/dev/tty.usbserial-*` on macOS, `/dev/ttyUSB*` on Linux, `COM*` on Windows)

The firmware idles after boot and prints `READY`. It captures only when the host sends a `START <n>` command, then streams exactly `n` reading lines and prints `DONE`. This makes each capture a bounded, scripted event rather than a free-running stream that has to be trimmed afterward.


## Capture Command

If a command-line capture process was used, include the exact command:

```bash
python3 scripts/capture_gesture.py --port /dev/tty.usbserial-0001 \
    --gesture circle --participant 1 --session 1 --reps 5 --samples 150
```

Explain:

- What the command does: it opens the serial port, and then for each of the `--reps` repetitions it waits for the participant to press Enter, counts down three seconds, sends `START 150` to the board, reads exactly 150 reading lines, and writes them to a CSV. Each repetition is a separate file. The command is run once per gesture class, changing only `--gesture`.
- Which device or serial port it uses: the `--port` argument names the USB-to-serial adapter wired to the board's USART2 pins (adapter RX to PA2, adapter TX to PA3). The ST-LINK's own USB connection is not used for data, it carries no accelerometer stream.
- How long each capture lasts: 150 samples at 50 Hz is a 3.0-second window. This was chosen to comfortably contain gestures that take one to two seconds, leaving a short lead-in before the motion and a tail after it.
- Where the resulting CSV is saved: `data/raw/<gesture>/p##_s##_<gesture>_###.csv`. The script creates the gesture directory if needed, lowercases the gesture name, the participant, session, and repetition fields. Each file is written to a temporary name and renamed into place only after the capture completes, so an interrupted capture cannot leave a truncated CSV behind.

---

# Data Collection Procedure

## Collection Protocol

Each participant followed the same basic data collection procedure.

1. Place or hold the sensor in the standardized orientation.
2. Prepare the recording software.
3. Begin data recording.
4. Start from the designated neutral position.
5. Perform the assigned gesture.
6. Return to the neutral position if required.
7. Stop recording.
8. Save the capture using the required filename convention.
9. Verify that the file was created successfully.
10. Repeat for the required number of repetitions.

Additional instructions:

- Perform all repetitions of one gesture consecutively before moving to the next class, so the gesture definition stays fresh and consistent across the five repetitions.
- Counts 1, 2, 3 and start the motion on "GO!" rather than before it, so the gesture falls inside the recorded window.
- Keep the board level throughout; do not tilt or rotate the wrist to follow the shape.
- Return to a still position at the end of the gesture and hold until the capture finishes.
- Use exactly the five class names `circle`, `square`, `triangle`, `wave`, `idle`. The capture script creates a new directory for any name it is given, so a typo silently produces an unintended class.
- Confirm after each session that `data/raw` contains exactly five directories and that each holds the expected number of files.

## Participants

Number of participants:

**5 participants**

Participant identifiers are anonymous and use the following format:

```text
p01
p02
p03
p04
p05
```

Each team member recorded as one participant. No personally identifying information is stored in the dataset. Participant numbers were assigned before collection so that two people could not record under the same identifier.

## Sessions

Each participant collected data across:

**1 sessions**

Sessions are identified using:

```text
s01
```

Describe the separation between sessions:

Every participant recorded a single session, so all files carry `s01`. The session field exists in the filename convention so that additional sessions can be added later without changing the format.
Variation in this dataset therefore comes from the five different participants rather than from any one participant recording on multiple occasions. Each participant recorded at a different time, on their own hardware setup, after independently flashing the firmware — so device orientation in the hand, gesture speed, and gesture amplitude vary between participants even though the session index does not.

## Repetitions

Each participant recorded:

**5 repetitions per gesture per session**

Expected captures per participant:

| Gesture | Session 1 | Total |
|---|---:|---:|---:|
| `circle` | 5 | 5 |
| `square` | 5 | 5 |
| `triangle` | 5 | 5 |
| `wave` | 5 | 5 |
| `idle` | 5 | 5 |
| Total | 25 | 25 |
---

# Variation During Collection

To make the dataset more representative, participants were instructed to preserve the same basic gesture definition while allowing natural variation between repetitions.

Variation may include:

- Different participants - five people, each with different hand size, arm length, and natural gesture speed
- Natural differences in gesture speed within the one-to-two-second guideline
- Natural differences in gesture amplitude around the nominal 30 cm shape size
- Slight differences in movement direction and in how precisely the traced shape closes
- Slight differences in sensor orientation, since the board is hand-held rather than mounted
- Natural differences in how strongly the device is moved

Participants were still required to follow the same core gesture definition so that class labels remained consistent.

This variation was intentionally included to reduce the likelihood that the dataset only represents one person's exact movement pattern.

---

# Dataset Size and Balance

The final raw dataset contains:

- **Total captures:** 125
- **Participants:** 5
- **Sessions:** 1 per participant
- **Gesture classes:** 5

Each capture is a 3.0-second window of 150 samples at 50 Hz, giving 18,750 raw
accelerometer samples per class and 93,750 across the dataset.

## Class Distribution

| Class | Number of Raw Captures |
|---|---:|
| `circle` | 25 |
| `square` | 25 |
| `triangle` | 25 |
| `wave` | 25 |
| `idle` | 25 |
| **Total** | **125** |

Describe the balance of the dataset:

The dataset is exactly balanced. Every class contains 25 raw recordings, made up of 5 repetitions from each of the 5 participants, so no class is over- or under-represented and no single participant contributes more examples of one gesture than another. 
Balance was designed in rather than corrected afterward: the collection protocol assigned every participant the same five classes and the same number of repetitions, and `scripts/validate_dataset.py` reports the per-class counts and warns if any class deviates more than 20% from the mean. The final validation run reports zero warnings.
The dataset size was chosen to provide repeated examples of every class across multiple participants and sessions while remaining appropriate for a small course-scale gesture-recognition demonstration.

The dataset size was chosen to provide repeated examples of every class across multiple participants while remaining appropriate for a small course-scale gesture-recognition demonstration. Augmentation expands the 125 raw recordings to 500 augmented recordings (four methods per file), which are stored separately and never replace the raw data.

---

# Repository Structure

```text
.
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── circle/
│   │   ├── square/
│   │   ├── triangle/
│   │   ├── wave/
│   │   └── idle/
│   │
│   └── augmented/
│
├── scripts/
│   ├── augment.py
│   ├── train_demo.py
│   └── validate_dataset.py
│
└── docs/
    └── capture_notes.md
```

## Directory Descriptions

- `data/raw/` contains the original accelerometer recordings.
- `data/augmented/` contains generated augmented samples.
- `scripts/augment.py` performs accelerometer-specific augmentation.
- `scripts/train_demo.py` demonstrates model training and evaluation.
- `scripts/validate_dataset.py` checks dataset formatting and integrity.
- `docs/` contains supporting team documentation.

Raw CSV files are stored directly in the repository and are not submitted only inside an archive.

---

# File Naming Convention

Raw captures use the following format:

```text
p##_s##_<gesture>_###.csv
```

Example:

```text
p01_s01_circle_001.csv
```

The fields represent:

| Field | Meaning |
|---|---|
| `p01` | Participant identifier |
| `s01` | Session identifier |
| `circle` | Gesture/class name |
| `001` | Repetition number |

Example path:

```text
data/raw/circle/p01_s01_circle_001.csv
```

---

# CSV Format

Each raw sensor recording is stored as a CSV file.

The expected format is:

```text
timestamp,accel_x,accel_y,accel_z
```

Example:

```text
timestamp,accel_x,accel_y,accel_z
0.000,0.03,-0.02,9.79
0.010,0.05,-0.01,9.81
0.020,0.10,0.04,9.76
```

## Column Definitions

| Column | Description | Units |
|---|---|---|
| `timestamp` | Time associated with each sensor sample | [seconds / milliseconds] |
| `accel_x` | X-axis acceleration | [m/s² / g] |
| `accel_y` | Y-axis acceleration | [m/s² / g] |
| `accel_z` | Z-axis acceleration | [m/s² / g] |

If additional columns are present, document them here.

---

# Label Representation

The authoritative class label for each recording is:

**The name of the recording's parent directory under `data/raw/`.**

For example:

```text
data/raw/circle/p01_s01_circle_001.csv
```

has the class label:

```text
circle
```

The gesture name is also included in the filename as a consistency check.

If the filename and directory name disagree, the file should be considered invalid and corrected before training rather than automatically assigning a conflicting label.

---

# Dataset Validation

Dataset integrity can be checked using:

```bash
python scripts/validate_dataset.py
```

The validation script is intended to check:

- Required CSV columns that are present: `timestamp`, `accel_x`, `accel_y`, `accel_z`
- Files are readable and non-empty
- No missing, non-numeric, or non-finite accelerometer values
- Filenames match the `p##_s##_<gesture>_###.csv` convention
- No duplicate participant/session/gesture/repetition combinations
- Number of captures
- Class distribution
- Dataset balance
- Class counts, participant counts, and session counts are reported

Example output:

```text
Summary
-------
CSV files checked   : 25
Valid files         : 25
Files with errors   : 0
Errors              : 0
Warnings            : 0
Shortest capture    : 150 rows
Longest capture     : 150 rows
Average capture     : 150.0 rows

Class counts
------------
circle   : 5
idle     : 5
square   : 5
triangle : 5
wave     : 5

RESULT: PASS
```

Participant and session identifiers are counted and reported, but are not verified against a roster — the script cannot tell a valid participant number from a mistyped one, only that the format is well-formed.

---

# Data Quality and Rejected Captures

During collection and validation, recordings with serious quality problems were rejected and re-recorded when possible.

Possible rejection criteria include:

- Missing sensor samples
- Malformed CSV structure
- Incorrect column names
- Incorrect gesture label
- Incorrect filename
- Recording stopped too early
- Recording contains the wrong gesture
- Corrupted or unreadable file
- Large portions of missing data

## Final Known Data Quality Issues

[State whether any known issues remain.]

Example:

> No known malformed CSV files remained after final validation.

or:

> Three recordings contain shorter-than-normal capture windows but were retained because the gesture itself was complete.

---

# Data Augmentation

Dataset augmentation is performed using:

```text
scripts/augment.py
```

## Purpose

The purpose of augmentation is to create additional realistic variations of the recorded accelerometer signals without changing their class labels.

Augmented recordings are stored separately from the original raw data.

Raw recordings are never overwritten.

## Augmentation Techniques

The augmentation pipeline may apply the following accelerometer-appropriate transformations.

### Gaussian Noise / Jitter

Gaussian noise/jitter is implemented using NumPy's `rng.uniform()` method to first randomly prescribe some noise "strength". We then generate a noise vector of the same size as our data collection sample using NumPy's `rng.normal()` method. This noise vector is added to our original data sample, and saved as a new augmented sample. Using this method, we can simulate normal sensor noise and invariability between data collection runs, creating a more diverse and varied data set to train our model on.

### Amplitude Scaling

Amplitude scaling takes advantage of NumPy's `rng.uniform()` method to randomly generate a number between 0.8 and 1.2. A data sample then gets multiplied by this number, creating a new scaled augmentation data sample. This simulates in our specific case, for example, differing sizes of our gestures (i.e. one person ends up drawing a circle that is slightly larger than the next). By scaling our data samples as an augmentation method, we can help further verify that our model is resistant to slight differences in the size of gestures completed. 

### Temporal Shift

Temporal shift augmentation is implemented in `augment.py` by first drawing a random integer shift *k* (via `rng.integers()`), bounded to ±10% of the sample length, and circularly shift the accelerometer columns by *k* positions using the `np.roll()` method. Because the roll wraps values around rather than discarding them, we overwrite the vacated end with our own padding. The original first sample gets padded to the beginning if *k* is positive, and the original last sample gets padded to the end if *k* is negative, so the output will never contain artificial wrap-around values.

Temporal shifting simulates the natural variance in reaction time between the AI starting to "listen" to the model, and the time in which the gesture actually starts. Not every capture has the motion of the gesture starting or ending at the same sample index, and training on shifted copies keeps the model from learning where in the window a gesture happens, so it relies more on the overall shape of the gesture itself. 

### Small 3D Rotation

Small 3D rotations are implemented by drawing 3 random angles (for each axis) via NumPy's `rng.uniform()` method, each bounded to ±10°. This gets converted to radians and used to build a rotation matrix for each axis, and get combined in to a single rotation. The accelerometer x/y/z vector at every time stamp is multiplied by this matrix, rotating the whole recording in 3D by a small fixed amount, leaving the overall shape of the gesture itself untouched. 

Due to the device being hand-held, there will naturally be small differences in rotation when gestures are made. Applying small 3D rotations to data helps simulate these small differences in the way different people hold the board, and even in the differences between samples run by the same person. This helps keep the model from overfitting to one exact sensor orientation. 

## Running Augmentation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Run the augmentation script:

```bash
python scripts/augment.py
```

By default, all 4 augmentation methods will be applied to all CSV files found at `data/raw/` and saved to `data/augmented/`. Augmented files will also be grouped by gesture, just as in `data/raw` (i.e. `data/augmented/circle`, for example). You can also specify individual files to augment, output path, and augmentation method(s), as follows: 

```bash
python scripts/augment.py --input <path_to_input_file>.csv --output <output_path> --methods <augmentation_methods>
```

> [!NOTE]
> When passing the `--input` arg, only paths to CSV files are accepted as arguments, and it will not recursively search a directory. If you'd like to pass all the files in a directory, use globbing e.g. `data/raw/circle/*.csv` for all circle gestures, for example. Globbing/paths are handled in `scripts/augment.py`, thus independent of terminal environment (e.g. this trick will still work in PowerShell).

For example, running the following script:  

```bash
python scripts/augment.py --input data/raw/circle/p01_s01_circle_001.csv data/raw/square/p02_s01_square_001.csv --output data/augmented/ --methods scale shift
```

will result in the following files to be written: 

```text
.
├── data/
│   ├── augmented/
│   │    ├── circle/
│   │    │    ├──  p01_s01_circle_001_scale.csv
│   │    │    └──  p01_s01_circle_001_shift.csv
│   │    │
│   │    └── square/
│   │         ├──  p02_s01_square_001_scale.csv
│   │         └──  p02_s01_square_001_shift.csv
```


The augmentation flags available are as follows:


| Method Arguments | Description |
| --- | --- |
| jitter | Gaussian noise applied via `rng.normal()` method |
| scale | Amplitude scaling | 
| shift | Temporal shift |
| rotate | Small 3D rotation | 

> [!Note]
> Using `python scripts/augment.py --help` will return usage on how to use `augment.py` for quick referencing. 

---

# Model Training Demonstration

The repository contains a simple machine learning demonstration in:

```text
scripts/train_demo.py
```

The purpose of this script is to verify that the collected accelerometer dataset can be successfully loaded, processed, and used in a machine learning pipeline.

The objective is to demonstrate dataset usability rather than to create a production-quality gesture-recognition system.

---

# Training Pipeline

The demonstration performs the following steps:

1. Discover gesture CSV files.
2. Load X, Y, and Z accelerometer values.
3. Determine the class label.
4. Preprocess the recordings.
5. Convert recordings into suitable model inputs or features.
6. Divide the dataset into training and testing sets.
7. Train a classifier.
8. Evaluate the classifier.
9. Print model performance metrics.

---

# Model

Model used:

**Random Forest classifier** (scikit-learn, 100 trees, `random_state=0`)

## Model Rationale

A Random Forest was chosen because it trains in seconds on a dataset of this size, needs no feature scaling or hyperparameter search, and gives a stable baseline with a fixed random seed. 
The purpose here is to show that the dataset loads and trains, not to maximize accuracy, so a model with few moving parts is the right choice.

Each recording is reduced to a fixed-length feature vector of per-axis summary statistics: 
mean, standard deviation, minimum, maximum, peak-to-peak range, and RMS, computed over each of the three axes and the signal magnitude. 
This makes every recording the same size regardless of its length, at the cost of discarding the order in which samples occurred — see Dataset Limitations.

---

# Train/Test Split

The split holds out **whole participants**. All recordings from a held-out participant go to the test set and none of them appear in training, so test accuracy measures whether a gesture can be recognized from a person the model has never seen.

- Split unit: participant
- Random seed: fixed (`random_state=0`), so the split is reproducible
- Augmented recordings derived from a held-out participant's captures are excluded from training entirely

That last point is the important one. Augmentation produces four near-copies of every raw recording. If a jittered copy of a test recording were left in the training set, the model would effectively be tested on data it had already seen and would report an inflated accuracy. `splitData()` in `scripts/train_demo.py` drops those copies rather than only separating the raw files.

If only one participant is present, the script falls back to splitting by recording and prints a notice that it has done so. The participant-level split is the one used for the reported results.

---

# Running the Training Demo

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python scripts/train_demo.py
```

If command-line arguments are required, document the exact final command here.

---

# Training Results

Final demonstration results:

| Metric | Result |
|---|---:|
| Training accuracy | [X%] |
| Test accuracy | [X%] |
| Precision | [Optional] |
| Recall | [Optional] |
| F1-score | [Optional] |

If a confusion matrix is generated, describe or include it here.

Model accuracy itself is not the primary purpose of this project. The training demonstration exists to show that the submitted dataset can be consumed by a working machine learning pipeline.

---

# Installation

## Requirements

Recommended Python version:

```text
Python [version]
```

Required Python packages are listed in:

```text
requirements.txt
```

## Linux / WSL

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Windows PowerShell

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# Reproducing the Project

A new user should be able to reproduce the software portion of the project using the following process.

## 1. Clone the Repository

```bash
git clone https://github.com/ColoradoEmbeddedAI-Fall2026/embeddedai-fall2026-lab2-dataset-group-9.git
```

Enter the repository:

```bash
cd embeddedai-fall2026-lab2-dataset-group-9
```

## 2. Create the Python Environment

```bash
python -m venv .venv
```

Activate the environment.

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux / WSL

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Validate the Raw Dataset

```bash
python scripts/validate_dataset.py
```

## 4. Generate Augmented Data

```bash
python scripts/augment.py
```

## 5. Train the Demonstration Model

```bash
python scripts/train_demo.py
```

---

# Dataset Reliability and Quality Control

Several steps were taken to improve dataset reliability.

- All participants followed the same basic capture protocol.
- Gesture definitions were established before collection.
- Consistent filenames were used.
- Consistent directory labels were used.
- The dataset was checked for malformed CSV files.
- Class counts were monitored.
- Multiple participants contributed data.
- Multiple capture sessions were used.
- Corrupt or invalid samples were re-recorded when possible.
- Raw recordings were preserved separately from augmented recordings.

---

# Dataset Limitations

Known limitations include:

- Limited number of participants
- Limited number of capture sessions
- Limited environmental variation
- Possible variation in sensor placement
- Limited number of repetitions
- Use of a single accelerometer hardware platform
- Similar gestures may produce overlapping motion patterns
- Dataset size is appropriate for a course demonstration but may not be sufficient for production use
- [Additional limitation]

These limitations should be considered when interpreting model performance or using the dataset outside the scope of this assignment.

---

# Contributions

## Tyler Pilat

Primary role:

**Project Infrastructure / Standards**

Responsibilities:

- Repository structure
- Dataset standards
- CSV format
- Filename convention
- Initial capture protocol
- Dataset validation script
- Initial README structure

## Logan Caskey

Primary role:

**Data Collection / Quality Assurance**

Responsibilities:

- Coordinate team data collection
- Track repetitions
- Track sessions
- Check class balance
- Identify missing recordings
- Identify invalid recordings
- Organize final raw dataset

## Logan Caskey

Primary role:

**Data Augmentation**

Responsibilities:

- Implement `augment.py`
- Select appropriate augmentation methods
- Preserve class labels during augmentation
- Test generated data
- Document augmentation behavior

## Julian Werder

Primary role:

**Model Training / Evaluation**

Responsibilities:

- Implement `train_demo.py`
- Load and preprocess the dataset
- Create training and testing sets
- Train the classifier
- Evaluate the classifier
- Report results

## Joel Ramos Silva & Jacob Thiel

Primary role:

**Documentation / Final Integration**

Responsibilities:

- Complete final README
- Document capture procedure
- Document augmentation and training scripts
- Maintain `requirements.txt`
- Verify installation instructions
- Perform final rubric review
- Perform final repository integration

All team members contributed accelerometer recordings to the raw dataset.

---

# Final Verification Checklist

Before submission, the team should verify:

- [ ] All required raw CSV files are included directly in the repository.
- [ ] Raw CSV files are not stored only inside a ZIP archive.
- [ ] All gesture classes are represented.
- [ ] Classes are reasonably balanced.
- [ ] Multiple repetitions exist for every gesture.
- [ ] Multiple participants and/or sessions are represented.
- [ ] Gesture definitions are clearly documented.
- [ ] Idle/background behavior is clearly documented.
- [ ] Sensor hardware is documented.
- [ ] Sensor placement is documented.
- [ ] Sampling rate and acceleration units are documented.
- [ ] Exact capture software or command is documented.
- [ ] CSV columns are documented.
- [ ] Authoritative label source is explicitly documented.
- [ ] Filename convention is documented.
- [ ] Dataset variation is documented.
- [ ] Data-quality rejection criteria are documented.
- [ ] All raw files pass final validation.
- [ ] `augment.py` executes successfully.
- [ ] Augmentation methods are documented.
- [ ] Augmented data does not overwrite raw data.
- [ ] `train_demo.py` executes successfully.
- [ ] Training and testing procedure is documented.
- [ ] Model results are documented.
- [ ] `requirements.txt` contains all required dependencies.
- [ ] Commands in this README were tested in a clean environment.
- [ ] Dataset  are documented.
- [ ] Repository structure is clean and understandable.
- [ ] Assignment self-check/bundle tool has been run.
- [ ] Final repository was reviewed against the assignment rubric.

---

# Course Use

This dataset was created for educational use as part of Embedded AI coursework at the University of Colorado Boulder.

[Add any additional licensing or course-use information if required.]
