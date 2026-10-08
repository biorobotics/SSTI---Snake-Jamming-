"""Five-module q=0.65 anchor; selected fifth joint is +pi/4."""
import argparse
from pathlib import Path
import sys

import run_mujoco

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tail',choices=['plus45','minus45'],default='plus45')
    parser.add_argument('--play',action='store_true')
    args,extra = parser.parse_known_args()
    scene = HERE.parent/'assets'/f'xml_anchor_q065_5modules_{args.tail}.xml'
    sys.argv = [str(HERE/'run_mujoco.py'),'--model',str(scene),'--keyframe','q065',
                '--output',str(HERE.parent/'outputs'/f'q065_5modules_{args.tail}')]
    if not args.play:
        sys.argv.append('--paused')
    sys.argv.extend(extra)
    run_mujoco.main()


if __name__=='__main__':
    main()
