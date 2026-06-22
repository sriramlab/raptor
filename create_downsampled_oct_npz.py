import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from embed import equidistant_subsample_volume, load_oct_path, save_name_for_pid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--saveto', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--many', type=int, default=None)
    parser.add_argument('--subsample_factor', type=int, default=4)
    parser.add_argument('--skip_existing', action='store_true', default=False)
    parser.add_argument('--continue_on_error', action='store_true', default=False)
    parser.add_argument('--errors_log', default=None)
    args = parser.parse_args()

    with open(args.manifest) as fl:
        paths = [ln.strip() for ln in fl if ln.strip()]
    if args.many is None:
        batch = paths[args.start:]
    else:
        batch = paths[args.start:args.start + args.many]

    outdir = Path(args.saveto)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.errors_log:
        Path(args.errors_log).parent.mkdir(parents=True, exist_ok=True)

    for path in tqdm(batch):
        outpath = outdir / f'{save_name_for_pid(path)}.npz'
        if args.skip_existing and outpath.exists():
            continue
        try:
            vol = load_oct_path(path)
            vol = equidistant_subsample_volume(vol, args.subsample_factor)
            np.savez_compressed(outpath, oct_volume_normalized=vol.astype(np.float16))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            msg = f'{path}\t{type(exc).__name__}\t{exc}'
            tqdm.write(msg)
            if args.errors_log:
                with open(args.errors_log, 'a') as err_fl:
                    err_fl.write(msg + '\n')


if __name__ == '__main__':
    main()
