"""
Script for creating sentence-embeddings from raw sentences.
INPUT: {lang}.txt
OUTPUT: {lang}.hdf5
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
import os
import time
import argparse
from itertools import (takewhile,repeat)

# from logger import log
from storage import Storage

def count_lines(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum( buf.count(b'\n') for buf in bufgen )

def get_line(f_path):
    with open(f_path) as file:
        for line in file:
            yield line


def dset_name_to_resume_idx(dset_name):
    return (int(dset_name.split("_")[-1]) + 1) * HK


def idx_to_dset_name(idx):
    return DSET_PREFIX.format(int(idx/HK)-1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument("-i", "--input", help = "Input sentences as lines of plaintext file", required = True)
    required.add_argument("-o", "--output", help = "Output hdf5 path", required = True)
    required.add_argument("-s", "--source_lang", help = "Source language identifier (eg. ta)", required = True)
    args = parser.parse_args()
    

    F_PATH = args.input
    OUT_PATH = args.output
    SOURCE_LANG = args.source_lang
    TOTAL_SIZE = count_lines(F_PATH)

    CHUNK_SIZE = 50000
    DSET_PREFIX = "{}_emds_{}".format(SOURCE_LANG, '{}')
    HK = 100000

    model = SentenceTransformer('sentence-transformers/LaBSE')

    pool = model.start_multi_process_pool()
    hk_embeddings, sentences = [], []
    hdf5_store = Storage(OUT_PATH)
    dataset_count = processed_count = skip_count = 0
    print("{} lines found".format(TOTAL_SIZE))
    print("Starting embedding creation.")
    gen = get_line(F_PATH)
    is_complete = False

    s_last_line = last_line = None
    while True:
        tm = time.time()
        CHUNK_SIZE = CHUNK_SIZE if (TOTAL_SIZE - processed_count) > CHUNK_SIZE else TOTAL_SIZE - processed_count
        sentences = [" ".join(next(gen).split()) for _ in range(CHUNK_SIZE)]
        if processed_count+CHUNK_SIZE == TOTAL_SIZE:
            print("Completed Reading All sentences from file.")
            is_complete = True

        embeddings = model.encode_multi_process(sentences,pool)
        hk_embeddings.extend(embeddings)
        processed_count += len(sentences)

        if len(hk_embeddings) >= HK:
            dset_name = DSET_PREFIX.format(dataset_count)
            hdf5_store.store(hk_embeddings[:HK], dset_name)
            hk_embeddings = hk_embeddings[HK:]
            dataset_count += 1
        print("100K Embeddings shape: {}".format(np.shape(hk_embeddings)))
        print("Processed {}/{} sentences successfully.".format(processed_count,TOTAL_SIZE))
        print("Time taken to add {} sentences is : {} secs".format(CHUNK_SIZE, time.time()-tm))
        if is_complete:
            break

    if hk_embeddings:
        dset_name = DSET_PREFIX.format(dataset_count)
        hdf5_store.store(hk_embeddings, dset_name)

    print("{} Embeddings computed and Stored successfully.".format(processed_count))
