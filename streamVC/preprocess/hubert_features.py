import logging
from hubert.dump_km_labels import dump_label

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir")
    parser.add_argument("split")
    parser.add_argument("km_path")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("lab_dir")
    args = parser.parse_args()
    logging.info(str(args))

    dump_label(**vars(args))