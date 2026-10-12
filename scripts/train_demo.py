#!/usr/bin/env python3
import argparse
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED = 0

class GestureRecording:
    def __init__(self, path: Path, label: str, data: np.ndarray, augmented: bool = False):
        self.path = path
        self.label = label
        self.data = data
        self.augmented = augmented
        # p01_s01_circle_001(_jitter) -> p01
        self.participant = path.stem.split("_")[0]
        # augmented files keep the raw file name + _method, so they can be traced back to their source
        self.source = path.stem.rsplit("_", 1)[0] if augmented else path.stem

class GestureTrainer:
    def __init__(self, raw_dir: str = "data/raw", aug_dir: str | None = "data/augmented",
                 test_size: float = 0.2, seed: int = SEED):
        self.raw_dir : Path = Path(raw_dir)
        self.aug_dir : Path | None = Path(aug_dir) if aug_dir is not None else None
        self.test_size : float = test_size
        self.seed : int = seed
        self.recordings : list[GestureRecording] = []
        self.model : RandomForestClassifier = RandomForestClassifier(n_estimators=100, random_state=seed)

    def loadFolder(self, folder: Path, augmented: bool) -> int:
        count = 0
        # the parent directory is the authoritative label
        for f in sorted(folder.rglob("*.csv")):
            try:
                data = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
            except ValueError as e:
                print(f"Skipping unreadable file {f}: {e}")
                continue
            if data.shape[0] == 0 or data.shape[1] < 4:
                print(f"Skipping empty or malformed file {f}")
                continue
            self.recordings.append(GestureRecording(f, f.parent.name, data, augmented))
            count += 1
        return count

    def loadDataset(self) -> bool:
        if not self.raw_dir.is_dir():
            print(f"Raw data directory not found: {self.raw_dir} (run from repo root or pass --raw-dir)")
            return False

        raw_count = self.loadFolder(self.raw_dir, False)
        aug_count = 0
        if self.aug_dir is not None and self.aug_dir.is_dir():
            aug_count = self.loadFolder(self.aug_dir, True)

        print(f"Loaded {raw_count} raw and {aug_count} augmented recordings")
        if raw_count == 0:
            print("No raw recordings found, nothing to train on")
            return False

        for label in sorted({r.label for r in self.recordings}):
            n_raw = sum(1 for r in self.recordings if r.label == label and not r.augmented)
            n_aug = sum(1 for r in self.recordings if r.label == label and r.augmented)
            print(f"  {label}: {n_raw} raw, {n_aug} augmented")
        return True

    def extractFeatures(self, recording: GestureRecording) -> np.ndarray:
        # recordings are different lengths, so turn each one into fixed size stats per axis
        accel = recording.data[:, 1:4]
        magnitude = np.linalg.norm(accel, axis=1, keepdims=True)
        signals = np.hstack((accel, magnitude))

        features = [
            signals.mean(axis=0),
            signals.std(axis=0),
            signals.min(axis=0),
            signals.max(axis=0),
            np.ptp(signals, axis=0),
            np.sqrt((signals ** 2).mean(axis=0)),  # RMS
        ]
        return np.concatenate(features)

    def splitData(self) -> tuple[list[GestureRecording], list[GestureRecording]]:
        rng = np.random.default_rng(self.seed)
        raw = [r for r in self.recordings if not r.augmented]
        participants = sorted({r.participant for r in raw})

        # hold out whole participants so the test set is people the model has never seen
        if len(participants) >= 2:
            n_test = max(1, round(len(participants) * self.test_size))
            test_groups = set(rng.permutation(participants)[:n_test])
            print(f"Splitting by participant, test participants: {', '.join(sorted(test_groups))}")
            in_test = lambda r: r.participant in test_groups
        else:
            # only one participant, fall back to splitting by raw file
            sources = sorted({r.source for r in raw})
            n_test = max(1, round(len(sources) * self.test_size))
            test_groups = set(rng.permutation(sources)[:n_test])
            print("Only one participant found, splitting by recording instead")
            in_test = lambda r: r.source in test_groups

        # augmented copies of test recordings are dropped so they can't leak into training
        train = [r for r in self.recordings if not in_test(r)]
        test = [r for r in raw if in_test(r)]
        return train, test

    def train(self) -> bool:
        train_set, test_set = self.splitData()
        if len(train_set) == 0 or len(test_set) == 0:
            print(f"Not enough data to split (train: {len(train_set)}, test: {len(test_set)})")
            return False
        print(f"Train recordings: {len(train_set)}, test recordings: {len(test_set)}")

        x_train = np.array([self.extractFeatures(r) for r in train_set])
        y_train = np.array([r.label for r in train_set])
        x_test = np.array([self.extractFeatures(r) for r in test_set])
        y_test = np.array([r.label for r in test_set])

        self.model.fit(x_train, y_train)
        self.evaluate(x_train, y_train, x_test, y_test)
        return True

    def evaluate(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        labels = sorted(set(y_train) | set(y_test))
        train_pred = self.model.predict(x_train)
        test_pred = self.model.predict(x_test)

        print("\nResults")
        print("-------")
        print(f"Training accuracy: {accuracy_score(y_train, train_pred) * 100:.1f}%")
        print(f"Test accuracy:     {accuracy_score(y_test, test_pred) * 100:.1f}%")

        print("\nClassification report (test set):")
        print(classification_report(y_test, test_pred, labels=labels, zero_division=0))

        print("Confusion matrix (rows = true, columns = predicted):")
        matrix = confusion_matrix(y_test, test_pred, labels=labels)
        width = max(len(l) for l in labels) + 2
        print(" " * width + "".join(f"{l:>{width}}" for l in labels))
        for label, row in zip(labels, matrix):
            print(f"{label:<{width}}" + "".join(f"{v:>{width}}" for v in row))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a demo gesture classifier on the dataset.")
    parser.add_argument("--raw-dir", default="data/raw", help="path to raw gesture CSVs")
    parser.add_argument("--aug-dir", default="data/augmented", help="path to augmented gesture CSVs")
    parser.add_argument("--no-augmented", action="store_true", help="train on raw recordings only")
    parser.add_argument("--test-size", type=float, default=0.2, help="fraction of participants held out for testing")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed for the split and model")
    args = parser.parse_args()

    trainer = GestureTrainer(args.raw_dir, None if args.no_augmented else args.aug_dir, args.test_size, args.seed)
    if trainer.loadDataset():
        trainer.train()
