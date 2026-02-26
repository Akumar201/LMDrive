#!/usr/bin/env python3
"""
Summarize and compare multiple LMDrive benchmark result JSON files.

Prints a table of DS / RC / IS across runs, plus an average row,
and a copy-paste markdown table for the README.

Usage:
  # All 8-bit tiny runs:
  python3 leaderboard/scripts/summarize_results.py --dir results/ --pattern "tiny_8bit_*.json"

  # Compare 4-bit vs 8-bit averages:
  python3 leaderboard/scripts/summarize_results.py --dir results/ --pattern "tiny_*.json"

  # Pass files directly:
  python3 leaderboard/scripts/summarize_results.py results/tiny_8bit_20260225_120000.json results/tiny_4bit_20260225_130000.json
"""

import argparse
import glob
import json
import os
import sys


def extract_metrics(json_path):
    """Parse a leaderboard result JSON and return a dict of key metrics."""
    with open(json_path) as f:
        data = json.load(f)

    labels = data.get('labels', [])
    values = data.get('values', [])
    progress = data.get('_checkpoint', {}).get('progress', [0, 0])

    lv = dict(zip(labels, values))

    def get(key):
        v = lv.get(key)
        return float(v) if v is not None else None

    # Labels from statistics_manager.py:
    #   'Avg. driving score', 'Avg. route completion', 'Avg. infraction penalty'
    return {
        'file': os.path.basename(json_path),
        'ds': get('Avg. driving score'),
        'rc': get('Avg. route completion'),
        'is_': get('Avg. infraction penalty'),
        'progress': '{}/{}'.format(progress[0], progress[1]) if len(progress) >= 2 else '?/?',
    }


def fmt(v):
    if isinstance(v, float):
        return '{:.3f}'.format(v)
    return 'N/A' if v is None else str(v)


def avg(lst):
    return sum(lst) / len(lst) if lst else None


def main():
    parser = argparse.ArgumentParser(
        description='Summarize LMDrive benchmark results across multiple runs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('files', nargs='*', help='Result JSON file paths')
    parser.add_argument('--dir', help='Directory to search for result files')
    parser.add_argument('--pattern', default='*.json',
                        help='Glob pattern within --dir (default: *.json)')
    args = parser.parse_args()

    files = list(args.files)
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, args.pattern)))

    if not files:
        print('No result files specified. Use --dir/--pattern or pass file paths directly.')
        sys.exit(1)

    rows = []
    for f in sorted(set(files)):
        if not os.path.exists(f):
            print('Warning: {} not found, skipping.'.format(f))
            continue
        try:
            rows.append(extract_metrics(f))
        except Exception as e:
            print('Warning: could not parse {}: {}'.format(f, e))

    if not rows:
        print('No valid result files parsed.')
        sys.exit(1)

    name_w = max(max(len(r['file']) for r in rows), 10)
    header = '{:<{w}}  {:>8}  {:>8}  {:>8}  {:>10}'.format(
        'Run', 'DS', 'RC (%)', 'IS', 'Progress', w=name_w)
    sep = '-' * len(header)

    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print('{:<{w}}  {:>8}  {:>8}  {:>8}  {:>10}'.format(
            r['file'], fmt(r['ds']), fmt(r['rc']), fmt(r['is_']), r['progress'], w=name_w))
    print(sep)

    valid_ds = [r['ds'] for r in rows if r['ds'] is not None]
    valid_rc = [r['rc'] for r in rows if r['rc'] is not None]
    valid_is = [r['is_'] for r in rows if r['is_'] is not None]
    n = len(rows)

    if valid_ds:
        a_ds, a_rc, a_is = avg(valid_ds), avg(valid_rc), avg(valid_is)
        print('{:<{w}}  {:>8}  {:>8}  {:>8}'.format(
            'AVERAGE ({} runs)'.format(n), fmt(a_ds), fmt(a_rc), fmt(a_is), w=name_w))
    print(sep)

    # Markdown table for copy-pasting into README
    print('\n### Markdown table\n')
    print('| Run | DS | RC (%) | IS | Progress |')
    print('|-----|----|--------|----|----------|')
    for r in rows:
        print('| {} | {} | {} | {} | {} |'.format(
            r['file'], fmt(r['ds']), fmt(r['rc']), fmt(r['is_']), r['progress']))
    if valid_ds:
        print('| **Average ({} runs)** | **{}** | **{}** | **{}** | |'.format(
            n, fmt(a_ds), fmt(a_rc), fmt(a_is)))


if __name__ == '__main__':
    main()
