"""Open a preserved-asset five-module version with direct torque motors."""
import argparse
from pathlib import Path
import sys

import run_mujoco

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collision',choices=['convex','proxy'],default='proxy')
    parser.add_argument('--play',action='store_true')
    args,extra = parser.parse_known_args()
    scene = HERE.parent/'assets'/f'xml_anchor_q065_5modules_torque_{args.collision}.xml'
    sys.argv = [str(HERE/'run_mujoco.py'),'--model',str(scene),'--keyframe','q065',
                '--torque-control','--output',str(HERE.parent/'outputs'/f'q065_5modules_torque_{args.collision}')]
    if not args.play:
        sys.argv.append('--paused')
    sys.argv.extend(extra)
    run_mujoco.main()


if __name__=='__main__':
    main()
