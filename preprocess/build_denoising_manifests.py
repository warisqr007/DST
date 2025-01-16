from glob import glob
import argparse
from collections import defaultdict, Counter
from itertools import combinations, product, groupby
from pathlib import Path
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import random
from shutil import copy
from subprocess import check_call

np.random.seed(42)
random.seed(42)


def get_fname(s):
    return s.split("\t")[0]

def get_emotion(s):
    return get_fname(s).split("_")[0].split("/")[1].lower()

def get_utt_id(s):
    return get_fname(s).split(".")[0].split("_")[-1]

def dedup(seq):
    """ >> remove_repetitions("1 2 2 3 100 2 2 1")
    '1 2 3 100 2 1' """
    seq = seq.strip().split(" ")
    result = seq[:1]
    reps = []
    rep_counter = 1
    for k in seq[1:]:
        if k != result[-1]:
            result += [k]
            reps += [rep_counter]
            rep_counter = 1
        else:
            rep_counter += 1
    reps += [rep_counter]
    assert len(reps) == len(result) and sum(reps) == len(seq)
    return " ".join(result) + "\n" #, reps

def remove_under_k(seq, k):
    """ remove tokens that repeat less then k times in a row
    >> remove_under_k("a a a a b c c c", 1) ==> a a a a c c c """
    seq = seq.strip().split(" ")
    result = []

    freqs = [(k,len(list(g))) for k, g in groupby(seq)]
    for c, f in freqs:
        if f > k:
            result += [c for _ in range(f)]
    return " ".join(result) + "\n" #, reps


def call(cmd):
    print(cmd)
    check_call(cmd, shell=True)


def denoising_preprocess(path, lang, dict):
    bin = 'fairseq-preprocess'
    cmd = [
        bin,
        f'--trainpref {path}/train.{lang} --validpref {path}/valid.{lang} --testpref {path}/test.{lang}',
        f'--destdir {path}/tokenized/{lang}',
        '--only-source',
        '--task denoising',
        '--workers 40',
    ]
    if dict != "":
        cmd += [f'--srcdict {dict}']
    cmd = " ".join(cmd)
    call(cmd)


def load_tokens(path):
    assert path.exists()
    tokens_lines = open(path, "r").readlines()
    root, tokens_lines = tokens_lines[0], tokens_lines[1:]
    path_lines = [l.split("|")[0] for l in tokens_lines]
    tokens_lines = [l.split("|")[1] for l in tokens_lines]
    return root, path_lines, tokens_lines


def main():
    desc = """
    this script takes as input .tsv and .km files for EMOV dataset, and a pairs of emotions.
    it generates parallel .tsv and .km files for these emotions. for exmaple:
    ❯ python build_denoising_manifests.py \
            data/american_libri/tokens.txt \
            ~/tmp/denoise_data \
            --dedup --shuffle
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("data", type=Path, help="path to a dir containing .txt files with the data")
    parser.add_argument("output_path", type=Path, help="output directory with the manifests will be created")
    parser.add_argument("-dd", "--dedup", action='store_true', help="remove repeated tokens (example: 'aaabc=>abc')", default=False)
    parser.add_argument("-sh", "--shuffle", action='store_true', help="shuffle the data", default=True)
    parser.add_argument("--dict", default="data/dict.txt", help="")
    args = parser.parse_args()

    suffix = ""
    if args.dedup: suffix += "_dedup"
    
    
    denoising_dir = Path(args.output_path) / ("libri_denoising" + suffix)
    os.makedirs(denoising_dir, exist_ok=True)

    root, tsv_lines, km_lines = load_tokens(path = args.data)

    # split the data
    tsv_lines_train, tsv_lines_test, km_lines_train, km_lines_test = train_test_split(tsv_lines, km_lines, test_size=0.01, random_state=42)
    tsv_lines_train, tsv_lines_valid, km_lines_train, km_lines_valid = train_test_split(tsv_lines_train, km_lines_train, test_size=0.01, random_state=42)

    for split in ["train", "valid", "test"]:
        if split == "train":
            tsv_lines, km_lines = tsv_lines_train, km_lines_train
        elif split == "valid":
            tsv_lines, km_lines = tsv_lines_valid, km_lines_valid
        elif split == "test":
            tsv_lines, km_lines = tsv_lines_test, km_lines_test
        else:
            raise ValueError("split should be 'train', 'valid', or 'test'")

        # generate data for the denoising task
        print("---")
        print("Split:", split)
        american_tsv, american_km = [], []
        for tsv_line, km_line in zip(tsv_lines, km_lines):
            km_line = km_line if not args.dedup else dedup(km_line)
            american_tsv.append(tsv_line)
            american_km.append(km_line)

        print(f"{len(american_tsv)} samples")
        open(denoising_dir / f"files.{split}.american", "w").writelines([root] + [f"{l}\n" for l in american_tsv])
        open(denoising_dir / f"{split}.american", "w").writelines(american_km)
        
        
    # fairseq-preprocess the denoising data
    denoising_preprocess(denoising_dir, "american", args.dict)
    os.system(f"cp {args.dict} {denoising_dir}/tokenized/dict.txt")


if __name__ == "__main__":
    main()