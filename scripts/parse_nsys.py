#!/usr/bin/env python3
import argparse, subprocess, sys, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rep', required=True, help='Path to .qdrep or .nsys-rep file')
    ap.add_argument('--out', required=True, help='CSV path to write kernel summary')
    args = ap.parse_args()

    cmd = ['nsys', 'stats', '--report', 'gpukernsum', '--format', 'csv', args.rep]
    print('[INFO] Running:', ' '.join(cmd))
    try:
        raw = subprocess.check_output(cmd, text=True)
    except Exception as e:
        print('[ERR] Failed to run nsys stats:', e)
        sys.exit(1)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(raw)
    print('[OK] Wrote', args.out)

if __name__ == '__main__':
    main()

