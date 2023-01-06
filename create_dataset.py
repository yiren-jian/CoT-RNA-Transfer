#### import from coconet
from pydca.contact_visualizer.contact_visualizer import DCAVisualizer
from pydca.fasta_reader import fasta_reader
from inputreader import InputReader

import subprocess
import numpy as np
import logging
import os, errno
import glob
from datetime import datetime
import pickle
import random
from pathlib import Path
from argparse import ArgumentParser
import sys

import torch
from torch.utils.data.dataset import Dataset


logger = logging.getLogger(__name__)


class CoCoNetDataset:
    """Implements RNA contact prediction using direct coupling analysis enhanced by
    a simple convolutional neural network.
    """

    def __init__(self, data_dir, linear_dist=None, contact_dist=None):
        """Initializes CocoNet instance.
        Parameters
        ----------
            self : CocoNet
                An instance of CocoNet class.
            dir_msa_files : str
                Path to directory containing the MSA files.
            dir_pdb_file : str
                Path to the directory containing the PDB files.
            dir_refseq_files : str
                Path to the directory containing reference sequence files.
            linear_dist : int
                Distance between sites in reference sequence
            contact_dist : float
                Maximum distance between two residues in PDB file to be considered
                contacts.
        """
        self.__data_dir = os.path.abspath(data_dir)
        self.__linear_dist = linear_dist if linear_dist is not None else 4
        self.__contact_dist = contact_dist if contact_dist is not None else 10.0

        pdb_chains_list_file = os.path.join(self.__data_dir, 'CCNListOfPDBChains.txt')
        msa_files_list_file = os.path.join(self.__data_dir, 'CCNListOfMSAFiles.txt')
        pdb_files_list_file = os.path.join(self.__data_dir, 'CCNListOfPDBFiles.txt')
        input_reader = InputReader()
        self.__msa_file_names_list = input_reader.read_from_one_column_text_file(msa_files_list_file)
        self.__pdb_chains_list = input_reader.read_from_one_column_text_file(pdb_chains_list_file)
        self.__pdb_file_names_list = input_reader.read_from_one_column_text_file(pdb_files_list_file)
        self.__msa_files_dir = os.path.join(self.__data_dir, 'MSA_200')        ######################### Using the MSA from the new directory
        self.__refseqs_dir = os.path.join(self.__data_dir, 'sequences')
        self.__pdb_files_dir = os.path.join(self.__data_dir, 'PDBFiles')
        self.__secstruct_files_dir = os.path.join(self.__data_dir, 'secstruct')
        self.msa_files_list  = [
            os.path.abspath(os.path.join(self.__msa_files_dir, msa_file + '.faclean')) for msa_file in self.__msa_file_names_list
        ]

        self.__refseqs = self.get_refseqs()
        self.__refseqs_len = self.get_refseqs_len()

        logmsg  = """
            Data directory          : {},
            PDB chains list file    : {},
            MSA files list file     : {},
            PDB files list file     : {},
        """.format(self.__data_dir, pdb_chains_list_file,
            msa_files_list_file, pdb_files_list_file,
        )

        logger.info(logmsg)
        return None


    @property
    def pdb_file_names_list(self):
        return self.__pdb_file_names_list

    @property
    def msa_file_names_list(self):
        return self.__msa_file_names_list

    @property
    def pdb_chains_list(self):
        return self.__pdb_chains_list


    def map_pdb_id_to_family(self):
        """Mapps PDB ID to family name.

        Parameters
        ----------
            self : CocoNet(self, data_dir, linear_dist=None, contact_dist=None)

        Returns
        -------
            pdb_id_to_fam_name : dict
                pdb_id_to_fam_name[pdb_id]=fam_name
        """
        pdb_id_to_fam_name = dict()
        for pdb_id, fam_name in zip(self.__pdb_file_names_list, self.__msa_file_names_list):
            pdb_id_to_fam_name[pdb_id] = fam_name
        return pdb_id_to_fam_name


    @staticmethod
    def _to_dict(files_list):
        """Puts a list of file paths into a dictionary

        Parameters
        ----------
            files_list : list
                A list of file paths

        Returns
        -------
            files_dict : dict
                A dictionary whose keys are basenames of files and values file path.
        """
        files_dict = dict()
        for f in files_list:
            basename, _ = os.path.splitext(os.path.basename(f))
            files_dict[basename] = f
        return files_dict


    def get_refseq_files_list(self):
        """
        """
        refseq_files_list = [
            os.path.join(self.__refseqs_dir, pdb_file[:4] + '.fa') for pdb_file in self.__pdb_file_names_list
        ]
        return tuple(refseq_files_list)


    def get_pdb_files_list(self):
        """
        """
        pdb_files_list = [
            os.path.join(self.__pdb_files_dir, pdb_file + '.pdb') for pdb_file in self.__pdb_file_names_list
        ]
        return tuple(pdb_files_list)


    def create_directories(self, dir_path):
        """Creates (nested) directory given path.

        Parameters
        ----------
            self : CocoNet
                An instance of CocoNet class.
            dir_path : str
                Directory path.

        Returns
        -------
            None : None
        """

        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno !=errno.EEXIST:
                logger.error('Unable to create directory using path {}'.format(
                    dir_path)
                )
                raise
        return None

    def get_pdb_data(self):
        """Computes mapped PDB contacts for multiple RNA families. The computed
        mapped PDB data is pickled and only recomputed if any of reference sequence
        files, PDB files or PDB chain metadata file is updated.

        Parameters
        ----------
            self : CocoNet
                An instance of CocoNet class

        Returns
        -------
            mapped_pdb_data : dict
                A dictionary whose keys are RNA familiy names and values dictionaries
                that have site pair keys and PDB data values.
        """
        refseq_files_list = self.get_refseq_files_list()
        pdb_files_list = self.get_pdb_files_list()

        logger.info('\n\tObtaining mapped PDB data')
        txtfreader = InputReader()
        mapped_pdb_data = dict()
        for chain_id, pdb_file, refseq_file, msa_file in zip(self.__pdb_chains_list, pdb_files_list, refseq_files_list, self.__msa_file_names_list):
            curr_pdb_data, _missing, _refseq_len = txtfreader.get_mapped_pdb_data(pdb_chain_id=chain_id,
                refseq_file=refseq_file, pdb_file=pdb_file, linear_dist=self.__linear_dist,
                contact_dist=self.__contact_dist
            )
            # self.__msa_file_names_list  contains the list of MSA files, not the full path of the files
            famname, _ext =  os.path.splitext(msa_file)
            mapped_pdb_data[famname] = curr_pdb_data
            print(famname)
        return mapped_pdb_data


    def get_refseqs(self):
        """Obtains reference sequences of several RNA famlies from fasta formatted file.

        Parameters
        ----------
            self : CocoNet
                An instance of CocoNet class

        Returns
        -------
            reference_sequenes_dict : dict()
            reference_sequences_dict[FAMILY_NAME] = sequence
        """

        refseq_files_list = self.get_refseq_files_list()
        logger.info('\n\tObtaining reference sequences from FASTA files')
        reference_sequences_dict = dict()
        for refseq_file, msa_file_basename in zip(refseq_files_list, self.__msa_file_names_list):
            # if reference sequence file contains multiple sequences, take the first one.
            reference_sequences_dict[msa_file_basename.strip()] = fasta_reader.get_alignment_from_fasta_file(refseq_file)[0].strip()
        return reference_sequences_dict


    def get_refseqs_len(self):
        """Obtains length of reference sequence for each RNA family

        Parameters
        ----------
            self : CocoNet
                An instance of CocoNet class

        Returns
        -------
            refseqs_len_dict : dict()
                refseqs_len_dict[FAMILY_NAME] = refseq_length
        """
        refseqs_dict = self.__refseqs
        logger.info('\n\tObtaining length of reference sequences for {} RNA families'.format(len(refseqs_dict)))

        refseqs_len_dict = {
            fam : len(refseqs_dict[fam]) for fam in refseqs_dict
        }
        return refseqs_len_dict


class RNA_DATASET(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, input_dict, label_dict):
        self.input_dict = input_dict
        self.label_dict = label_dict

        assert [k for k in self.input_dict] == [k for k in self.label_dict]
        self.rna_fam_names = [k for k in self.input_dict]
        self.data_len = len(self.rna_fam_names)

    def __getitem__(self, index):
        rna_fam_name = self.rna_fam_names[index]
        input_tensor = self.input_dict[rna_fam_name]
        label_tensor = self.label_dict[rna_fam_name]

        seq, msa, idx = input_tensor['seq'].squeeze(0), input_tensor['msa'].squeeze(0), input_tensor['index'].squeeze(0)
        return seq, msa, idx, label_tensor.squeeze(0)

    def __len__(self):
        return self.data_len


class RNA_FEATSET(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, input_dict, label_dict):
        self.input_dict = input_dict
        self.label_dict = label_dict

        assert [k for k in self.input_dict] == [k for k in self.label_dict]
        self.rna_fam_names = [k for k in self.input_dict]
        self.data_len = len(self.rna_fam_names)

    def __getitem__(self, index):
        rna_fam_name = self.rna_fam_names[index]
        input_tensor = self.input_dict[rna_fam_name]
        label_tensor = self.label_dict[rna_fam_name]

        return input_tensor.squeeze(0), label_tensor.squeeze(0)

    def __len__(self):
        return self.data_len


def pad_collate(batch):
    (xx, yy) = zip(*batch)    ## xx.shape[L, L, C], yy.shape[L, L]
    x_lens = [x.shape[0] for x in xx]
    y_lens = [y.shape[0] for y in yy]
    assert max(x_lens) == max(y_lens)
    max_L = max(x_lens)
    batch_size = len(x_lens)
    C = batch[0][0].shape[-1]

    xx_pad = torch.zeros(batch_size, max_L, max_L, C)
    yy_pad = torch.ones(batch_size, max_L, max_L) * (-100)    #### ignored index in nn.CrossEntropyLoss()

    for i in range(batch_size):
        xx_pad[i, :x_lens[i], :x_lens[i], :] = xx[i]
        yy_pad[i, :y_lens[i], :y_lens[i]] = yy[i]

    return xx_pad.float(), yy_pad.long()
