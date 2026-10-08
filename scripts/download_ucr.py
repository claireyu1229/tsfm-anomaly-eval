#!/usr/bin/env python3
"""Download the UCR Time Series Anomaly Archive (KDD Cup 2021).

The archive is not redistributed in this repository. This script fetches it
from the official UCR page and extracts the 250 ``UCR_Anomaly_FullData`` files
into ``data/UCR_Anomaly_FullData/``.

Source: Wu & Keogh, "Current Time Series Anomaly Detection Benchmarks are
Flawed and are Creating the Illusion of Progress", IEEE TKDE 2021.
https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from tsfm_anomaly.preprocess import UCR_FILENAME_PATTERN

URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"
EXPECTED_FILES = 250


def extract_full_data(zip_path: Path, target_dir: Path) -> int:
    """Copy only the ``UCR_Anomaly_FullData/*.txt`` members, flattened."""
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            name = Path(member).name
            # Use only the file name so no member can write outside target_dir.
            if "/UCR_Anomaly_FullData/" not in member or not UCR_FILENAME_PATTERN.match(name):
                continue
            (target_dir / name).write_bytes(archive.read(member))
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--zip", type=Path, help="use an already downloaded archive")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    zip_path = args.zip or args.data_dir / "UCR_TimeSeriesAnomalyDatasets2021.zip"
    if not zip_path.exists():
        args.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} (about 180 MB)")
        urllib.request.urlretrieve(URL, zip_path)

    target = args.data_dir / "UCR_Anomaly_FullData"
    count = extract_full_data(zip_path, target)
    print(f"Extracted {count} files to {target}")
    if count != EXPECTED_FILES:
        raise SystemExit(f"Expected {EXPECTED_FILES} files; the archive layout may have changed.")
    if args.zip is None and not args.keep_zip:
        zip_path.unlink()


if __name__ == "__main__":
    main()
