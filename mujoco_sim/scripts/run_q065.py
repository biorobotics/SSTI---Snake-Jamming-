"""Open the q=0.65 wall-formula scene using the existing baseline runner."""
import argparse
from pathlib import Path
import sys

import run_mujoco

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side',choices=['left','right'],default='left')
    parser.add_argument('--play',action='store_true',help='start passive dynamics instead of paused inspection')
    args, extra = parser.parse_known_args()
    scene = HERE.parent / 'assets' / f'xml_anchor_q065_{args.side}.xml'
    sys.argv = [str(HERE/'run_mujoco.py'),'--model',str(scene),'--keyframe','q065',
                '--output',str(HERE.parent/'outputs'/f'q065_{args.side}')]
    if not args.play:
        sys.argv.append('--paused')
    sys.argv.extend(extra)
    run_mujoco.main()


if __name__ == '__main__':
    main()
