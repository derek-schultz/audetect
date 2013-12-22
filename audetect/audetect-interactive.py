#!/usr/bin/env python
import argparse
import cv2

from audetect.detect import ActionUnitDetector
from audetect.train import FacialFeatureTrainer, ActionUnitTrainer
from audetect.utils import *

def main():
    args = parse_args()

    if args.data:
        paths.DATA_PATH = args.data

    if args.action == 'detect':
        detector = ActionUnitDetector(sequence=args.sequence,
                                      verbose=args.verbose)
        active_aus = detector.detect()
        print active_aus
        for image in detector.sequence:
            cv2.imshow("Sequence", image)
            cv2.waitKey(100)

    elif args.action == 'train':
        if not args.samples:
            samples = -1
        else:
            samples = args.samples
        if args.features:
            trainer = FacialFeatureTrainer(train_size=int(samples),
                                           verbose=args.verbose)
        elif args.action_units:
            trainer = ActionUnitTrainer(train_size=int(samples),
                                        verbose=args.verbose)

def parse_args():
    parser = argparse.ArgumentParser(description='Detects facial action units')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--data', metavar="DATA_PATH")
    subparsers = parser.add_subparsers(help='choose one', dest='action')

    detect_parser = subparsers.add_parser('detect',
                                          help='detect -h for more info')
    detect_parser.add_argument('--sequence', metavar="IMG_SEQUENCE_PATH",
                               required=False)

    train_parser = subparsers.add_parser('train',
                                         help='train -h for more info')
    group = train_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--features', action='store_true',
                       help="train facial feature data")
    group.add_argument('--action_units', action='store_true',
                       help="train action unit data")
    train_parser.add_argument('--samples', metavar="N", required=False)

    return parser.parse_args()

if __name__ == "__main__":
    main()