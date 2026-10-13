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
- Direction of motion: Clockwise, seen by the team memeber.
- Approximate motion size: About 30 cm in diameter
- End position: Back at the 12 o'clock starting point, one full loop completed, device held still.
- Any important restrictions:One loop only, completed in roughly one to two seconds. Smooth continuous speed — no stopping partway around. The board stays flat; do not rotate the wrist to follow the curve.

### `square`

Include:

- Starting position: Board flat and level at chest/neck height, at the top-left corner of the square
- Direction of motion: Clockwise — top-left edge, top right edge, bottom right edge, bottom left edge
- Approximate motion size: About 30 cm per side
- End position: Back at the top-left corner, device held still.
- Any important restrictions: One lap only, about two seconds total. Pause very briefly (roughly a tenth of a second) at each corners and move in straight lines between them. The pauses are
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
- Any important restrictions: 4–6 passes total within about two seconds, at a steady rhythm. Keep the board flat and level throughout, its an arm motion not a wrist flick or a tilt.


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
- **Sampling rate:** 50 Hz nominal (20 ms firmware print interval); about 46 Hz as recorded, since consecutive timestamps step roughly 21–22 ms apart. Timestamps in each CSV are the measured times, so the actual rate can be recovered per file.
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
    --gesture circle --participant 1 --session 1 --reps 5 --samples 100
```

Explain:

- What the command does: it opens the serial port, and then for each of the `--reps` repetitions it waits for the participant to press Enter, counts down three seconds, sends `START 100` to the board, reads exactly 100 reading lines, and writes them to a CSV. Each repetition is a separate file. The command is run once per gesture class, changing only `--gesture`.
- Which device or serial port it uses: the `--port` argument names the USB-to-serial adapter wired to the board's USART2 pins (adapter RX to PA2, adapter TX to PA3). The ST-LINK's own USB connection is not used for data, it carries no accelerometer stream.
- How long each capture lasts: 100 samples is a 2.0-second window at the nominal 50 Hz; at the measured rate the recorded captures span about 2.1–2.2 seconds. This was chosen to comfortably contain gestures that take one to two seconds, leaving a short lead-in before the motion and a tail after it.
- Where the resulting CSV is saved: `data/raw/<gesture>/p##_s##_<gesture>_###.csv`. The script creates the gesture directory if needed, lowercases the gesture name, the participant, session, and repetition fields. Each file is written to a temporary name and renamed into place only after the capture completes, so an interrupted capture cannot leave a truncated CSV behind.

If a phone application was used instead, describe the exact process for:

1. Starting a recording
2. Performing the gesture
3. Stopping the recording
4. Exporting the accelerometer data
5. Saving or renaming the CSV file

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

- [Instruction]
- [Instruction]
- [Instruction]

## Participants

Number of participants:

**[X participants]**

Participant identifiers are anonymous and use the following format:

```text
p01
p02
p03
p04
p05
```

No personally identifying information is stored in the dataset.

## Sessions

Each participant collected data across:

**[X sessions]**

Sessions are identified using:

```text
s01
s02
```

Describe the separation between sessions:

[For example: Sessions were recorded at different times of day, on different days, or after resetting the device.]

## Repetitions

Each participant recorded approximately:

**[X repetitions per gesture per session]**

Expected captures per participant:

| Gesture | Session 1 | Session 2 | Total |
|---|---:|---:|---:|
| `[gesture_1]` | [X] | [X] | [X] |
| `[gesture_2]` | [X] | [X] | [X] |
| `[gesture_3]` | [X] | [X] | [X] |
| `[gesture_4]` | [X] | [X] | [X] |
| `idle` | [X] | [X] | [X] |

---

# Variation During Collection

To make the dataset more representative, participants were instructed to preserve the same basic gesture definition while allowing natural variation between repetitions.

Variation may include:

- Different participants
- Multiple recording sessions
- Natural differences in gesture speed
- Natural differences in gesture amplitude
- Slight differences in movement direction
- Slight differences in sensor orientation
- Natural differences in how strongly the device is moved
- [Other intentionally collected variation]

Participants were still required to follow the same core gesture definition so that class labels remained consistent.

This variation was intentionally included to reduce the likelihood that the dataset only represents one person's exact movement pattern.

---

# Dataset Size and Balance

The final raw dataset contains:

- **Total captures:** [X]
- **Participants:** [X]
- **Sessions:** [X]
- **Gesture classes:** [X]

## Class Distribution

| Class | Number of Raw Captures |
|---|---:|
| `[gesture_1]` | [X] |
| `[gesture_2]` | [X] |
| `[gesture_3]` | [X] |
| `[gesture_4]` | [X] |
| `idle` | [X] |
| **Total** | **[X]** |

Describe the balance of the dataset:

[For example: The dataset was designed so that each gesture class contains approximately the same number of recordings.]

The dataset size was chosen to provide repeated examples of every class across multiple participants and sessions while remaining appropriate for a small course-scale gesture-recognition demonstration.

---

# Repository Structure

```text
.
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── [gesture_1]/
│   │   ├── [gesture_2]/
│   │   ├── [gesture_3]/
│   │   ├── [gesture_4]/
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

- Required CSV columns
- Empty files
- Unreadable files
- Missing accelerometer values
- Invalid filename formatting
- Gesture labels
- Participant identifiers
- Session identifiers
- Number of captures
- Class distribution
- Dataset balance
- [Any additional implemented checks]

Example output:

```text
Dataset Validation
------------------
Files checked: XXX
Valid files: XXX
Invalid files: 0

Class counts:
gesture_1: XX
gesture_2: XX
gesture_3: XX
gesture_4: XX
idle: XX
```

Update this section so it only claims checks that are actually implemented by the final validation script.

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
- [Other criteria used by the team]

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

[Describe implementation.]

Example purpose:

> Small Gaussian noise simulates normal sensor measurement noise and small variations between recordings.

### Amplitude Scaling

[Describe implementation.]

Example purpose:

> Scaling the acceleration magnitude simulates natural differences in how strongly or quickly participants perform a gesture.

### Temporal Shift

[Describe implementation.]

Example purpose:

> Temporal shifting changes where the gesture occurs within the recording window while preserving the gesture itself.

### Small 3D Rotation

[Describe implementation if used.]

Example purpose:

> Small rotations simulate realistic differences in device orientation between participants and recordings.

## Why These Augmentations Were Chosen

[Explain why the transformations are appropriate for accelerometer data.]

Example:

> These augmentations were selected because they represent realistic variations that could occur during gesture performance or sensor measurement. The transformations are kept small enough that the underlying gesture class should remain unchanged.

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
| jitter | Gaussian noise applied via np.random.normal() method |
| scale | Amplitude scaling | 
| shift | Temporal shift |
| rotate | Small 3D rotation | 

> ![Note]
> Using `python augment.py --help` will return usage on how to use `augment.py` for quick referencing. 

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

**[Model type]**

Possible examples include:

- Multilayer Perceptron
- Random Forest
- Logistic Regression
- Support Vector Machine
- Small 1D CNN

## Model Rationale

[Briefly explain why the model was selected.]

Example:

> A Random Forest classifier was chosen because it provides a simple baseline that can train quickly on a relatively small gesture dataset.

---

# Train/Test Split

Describe how training and testing samples are separated.

Include:

- Training percentage
- Testing percentage
- Random seed if used
- Whether the split is performed by file, session, or participant

Example:

> The dataset was divided into 80% training data and 20% testing data using a fixed random seed.

If possible, consider splitting by participant or session to reduce the risk of nearly identical recordings from the same collection session appearing in both training and testing sets.

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
- [Additional team procedure]

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

## [Name]

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

## [Name]

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

## [Name]

Primary role:

**Data Augmentation**

Responsibilities:

- Implement `augment.py`
- Select appropriate augmentation methods
- Preserve class labels during augmentation
- Test generated data
- Document augmentation behavior

## [Name]

Primary role:

**Model Training / Evaluation**

Responsibilities:

- Implement `train_demo.py`
- Load and preprocess the dataset
- Create training and testing sets
- Train the classifier
- Evaluate the classifier
- Report results

## [Name]

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
- [ ] Dataset limitations are documented.
- [ ] Repository structure is clean and understandable.
- [ ] Assignment self-check/bundle tool has been run.
- [ ] Final repository was reviewed against the assignment rubric.

---

# Course Use

This dataset was created for educational use as part of Embedded AI coursework at the University of Colorado Boulder.

[Add any additional licensing or course-use information if required.]
