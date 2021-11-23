def main(parser):
    parser.add_argument('-t',"--type", type=str,help='type of task',required=True)
    parser.add_argument('-f',"--folder", type=str,help='data folder',required=True)
    args, unknown = parser.parse_known_args()

    if args.type == "nmr":
        from autochem.spectra.nmr import cmd as nmrcmd
        nmrcmd.main(parser)
    else:
        raise NotImplementedError(f'unknown type {args.type}')



if __name__ == '__main__':
    import sys,os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import argparse
    parser = argparse.ArgumentParser(description='autochem command line')
    main(parser)