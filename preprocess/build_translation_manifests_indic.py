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
    return s.split("/")[-1].split(".")[0]

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



def translation_preprocess(path, src_lang, trg_lang, dict, only_train=False):
    bin = 'fairseq-preprocess'
    cmd = [
        bin,
        f'--source-lang {src_lang} --target-lang {trg_lang}',
        f'--trainpref {path}/train',
        f'--destdir {path}/tokenized',
        '--workers 40',
    ]
    if not only_train:
        cmd += [f'--validpref {path}/valid --testpref {path}/test']
    if dict != "":
        cmd += [
            f'--srcdict {dict}',
            f'--tgtdict {dict}',
        ]
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
    ❯ python build_translation_manifests.py \
            data/spanish/tokens.txt \
            data/american_arctic/tokens.txt \
            ~/tmp/emov_pairs \
            --dedup --shuffle --dry-run
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("src_data", type=Path, help="path to a dir containing .txt files with the data")
    parser.add_argument("tgt_data", type=Path, help="path to a dir containing .txt files with the data")
    parser.add_argument("output_path", type=Path, help="output directory with the manifests will be created")
    parser.add_argument("-dd", "--dedup", action='store_true', help="remove repeated tokens (example: 'aaabc=>abc')")
    parser.add_argument("-sh", "--shuffle", action='store_true', help="shuffle the data")
    parser.add_argument("-ae", "--autoencode", action='store_true', help="include training pairs from the same emotion (this includes examples of the same sentence uttered by different people and examples where the src and trg are the exact same seq)")
    parser.add_argument("-dr", "--dry-run", action='store_true', help="don't write anything to disk", default=False)
    parser.add_argument("--dict", default="data/dict.txt", help="")
    args = parser.parse_args()
    
    ACCENTS = ["indian", "american"]
    SRC_ACCENT = "indian"
    TRG_ACCENT = "american"

    suffix = ""
    if args.dedup: suffix += "_dedup"
    translation_suffix = ""
    if args.autoencode: translation_suffix += "_autoencode"

    translation_dir = Path(args.output_path) / ("translation" + f"_src_{SRC_ACCENT}" + suffix + translation_suffix)
    os.makedirs(translation_dir, exist_ok=True)
    

    ####### SPLIT THE DATA
    src_root, src_tsv_lines, src_km_lines = load_tokens(path = args.src_data)
    tgt_root, tgt_tsv_lines, tgt_km_lines = load_tokens(path = args.tgt_data)

    assert len(src_tsv_lines) == len(tgt_tsv_lines)
    # length = min(len(src_tsv_lines), len(tgt_tsv_lines))
    # src_tsv_lines = src_tsv_lines[:length]
    # tgt_tsv_lines = tgt_tsv_lines[:length]
    # src_km_lines = src_km_lines[:length]
    # tgt_km_lines = tgt_km_lines[:length]

    index_list = list(range(len(src_tsv_lines)))
    # Split the data
    index_list_train, index_list_test = train_test_split(index_list, test_size=0.01, random_state=42)
    index_list_train, index_list_valid = train_test_split(index_list_train, test_size=0.01, random_state=42)


    for split, indices in zip(["train", "valid", "test"], [index_list_train, index_list_valid, index_list_test]):
        print("---")
        print(split)

        src_tsv = [src_tsv_lines[i] for i in indices]
        tgt_tsv = [tgt_tsv_lines[i] for i in indices]
        src_km = [src_km_lines[i] for i in indices]
        tgt_km = [tgt_km_lines[i] for i in indices]

        # autoencode target
        if args.autoencode:
            src_tsv += tgt_tsv
            tgt_tsv += tgt_tsv
            src_km += tgt_km
            tgt_km += tgt_km

        if args.dedup:
            src_km = [dedup(l) for l in src_km]
            tgt_km = [dedup(l) for l in tgt_km]

        
        assert len(src_tsv) == len(tgt_tsv) == len(src_km) == len(tgt_km)
        print(f"{len(src_tsv)} pairs")

        if len(src_tsv) == 0:
            raise Exception("ERROR: generated 0 pairs!")

        if args.shuffle:
            src_tsv, tgt_tsv, src_km, tgt_km = shuffle(src_tsv, tgt_tsv, src_km, tgt_km, random_state=42)

        if args.dry_run: continue

        # create files
        os.makedirs(translation_dir / f"{SRC_ACCENT}-{TRG_ACCENT}", exist_ok=True)
        open(translation_dir / f"{SRC_ACCENT}-{TRG_ACCENT}" / f"files.{split}.{SRC_ACCENT}", "w").writelines([src_root] + [f"{l}\n" for l in src_tsv])
        open(translation_dir / f"{SRC_ACCENT}-{TRG_ACCENT}" / f"files.{split}.{TRG_ACCENT}", "w").writelines([tgt_root] + [f"{l}\n" for l in tgt_tsv])
        open(translation_dir / f"{SRC_ACCENT}-{TRG_ACCENT}" / f"{split}.{SRC_ACCENT}", "w").writelines(src_km)
        open(translation_dir / f"{SRC_ACCENT}-{TRG_ACCENT}" / f"{split}.{TRG_ACCENT}", "w").writelines(tgt_km)


    # fairseq-preprocess the translation data
    os.makedirs(translation_dir / "tokenized", exist_ok=True)
    translation_preprocess(translation_dir / f"{SRC_ACCENT}-{TRG_ACCENT}", SRC_ACCENT, TRG_ACCENT, args.dict)#, only_train=SRC_EMOTION==TRG_EMOTION)
    os.system(f"cp -rf {translation_dir}/**/tokenized/* {translation_dir}/tokenized")

if __name__ == "__main__":
    main()