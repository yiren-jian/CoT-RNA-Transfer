#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess


def msa_from_rna_to_protein(msa, max_seqs=200, AminoAcids='ACDE', save=False, old_path=None, new_path=None):
    lines = []
    for line in open(os.path.join(old_path, msa)):
        line = line.strip()
        if not line.startswith(">"):
            new_line = ''
            for l in line:
                if l == 'A':
                    new_line += AminoAcids[0]
                elif l == 'U':
                    new_line += AminoAcids[1]
                elif l == 'C':
                    new_line += AminoAcids[2]
                elif l == 'G':
                    new_line += AminoAcids[3]
                else:
                    new_line += '-'
            lines.append(new_line)
        else:
            lines.append(line)

    if max_seqs is not None:
        lines = lines[:2*max_seqs]     ### 2x for name and sequence

    if save:
        with open(os.path.join(new_path, msa), 'w') as f:
            for line in lines:
                f.write(f"{line}\n")


def reduce_msa_by_max_num(msa, max_seqs=200, save=False, old_path=None, new_path=None):
    lines = []
    for line in open(os.path.join(old_path, msa)):
        line = line.strip()
        lines.append(line)

    if max_seqs is not None:
        lines = lines[:2*max_seqs]     ### 2x for name and sequence

    if save:
        with open(os.path.join(new_path, msa), 'w') as f:
            for line in lines:
                f.write(f"{line}\n")


if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--AminoAcids', default='ACDE', type=str)
    args = parser.parse_args()

    assert len(args.AminoAcids) == 4

    old_path = 'RNA_DATASET/MSA/'
    msa_files = [f for f in listdir(old_path) if isfile(join(old_path, f))]
    new_path = 'RNA_DATASET/MSA_200/'
    if os.path.isdir(new_path) == False:
        os.makedirs(new_path)

    for msa in msa_files:
        msa_from_rna_to_protein(msa, max_seqs=200, AminoAcids=args.AminoAcids, save=True, old_path=old_path, new_path=new_path)


    old_path = 'RNA_TESTSET/MSA/'
    msa_files = [f for f in listdir(old_path) if isfile(join(old_path, f))]
    new_path = 'RNA_TESTSET/MSA_200/'
    if os.path.isdir(new_path) == False:
        os.makedirs(new_path)

    for msa in msa_files:
        msa_from_rna_to_protein(msa, max_seqs=200, AminoAcids=args.AminoAcids, save=True, old_path=old_path, new_path=new_path)

    old_path = 'RNA_DATASET/MSA/'
    msa_files = [f for f in listdir(old_path) if isfile(join(old_path, f))]
    new_path = 'RNA_DATASET/MSA_pydca/'
    if os.path.isdir(new_path) == False:
        os.makedirs(new_path)

    for msa in msa_files:
        reduce_msa_by_max_num(msa, max_seqs=200, save=True, old_path=old_path, new_path=new_path)


    old_path = 'RNA_TESTSET/MSA/'
    msa_files = [f for f in listdir(old_path) if isfile(join(old_path, f))]
    new_path = 'RNA_TESTSET/MSA_pydca/'
    if os.path.isdir(new_path) == False:
        os.makedirs(new_path)

    for msa in msa_files:
        reduce_msa_by_max_num(msa, max_seqs=200, save=True, old_path=old_path, new_path=new_path)
