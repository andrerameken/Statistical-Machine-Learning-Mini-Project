"""
Course: Statistical Machine Learning.
This script checks the predictions csv file for the most common formatting mistakes.
"""

import argparse
import collections
import sys

N_TEST_CASES = 387


def parse_args():
    """Parse and return command line arguments."""

    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--file",
        default="predictions.csv",
        required=False,
        help="Path to a CSV file with predictions. Default: predictions.csv")

    args = parser.parse_args()
    return args


def import_file(path):
    """Import the files as a list of lines"""
    print(f"Importing {path}...")
    try:
        with open(path, "rt") as f:
            data = f.readlines()
    except:
        print(f"File {path} not found. Exiting.")
        sys.exit(1)

    return data


def parse_and_check_(lines):
    """All prediction are expected to be 0/1 values in the first line separated by commas."""
    predictions = lines[0].strip().split(",")

    # Calculate frequencies.
    freq = collections.Counter(predictions)
    n = sum(freq.values())

    # Check the number of elements.
    if n != N_TEST_CASES:
        print(f"Error: the number of predictions must be {N_TEST_CASES}. Got {n}.")
        sys.exit(2)

    # Check the values.
    if set(predictions) != set(["0","1"]):
        print(f"Error: predicted values must be 0 or 1. Got {set(predictions)}.")
        sys.exit(3)

    print(f"The format seems to be correct. Your predicted frequencies: {freq}. Total number of predictions: {n}.")

    return predictions


def main():
    # Parse arguments.
    args = parse_args()

    # Import predictions.
    lines = import_file(args.file)

    # Parse and check predictions.
    parse_and_check_(lines)


if __name__ == '__main__':
    main()
