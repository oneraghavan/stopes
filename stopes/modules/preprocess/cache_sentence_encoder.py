# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp
import numpy as np
import h5py

from stopes.modules.preprocess.encode_to_npy import EncodeToNPY
from stopes.utils.mining_utils import extract_shard_id


class CacheTextEncoder(EncodeToNPY):
    """
    1. load a pre-trained model located in the HuggingFace sentence transformers Hub
    2. tokenize and encode input
    3. send embeddings to specified output file
    """

    def __init__(
        self,
        encoder_model: str,
        input_file: str,
        _name: str = "hf",
        output_dir: str = ".",
        input_file_idx: int = 0,
        outfile_prefix: str = "encf",
        outfile_postfix: str = "",
        normalize: bool = False,
        fp16_storage: bool = False,
        cached_embeddings: str = "",
        # ignored
        spm_vocab: str = "",
        spm_model: str = "",
    ) -> None:
        super().__init__(
            outfile_prefix=outfile_prefix,
            outfile_postfix=outfile_postfix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
            normalize=normalize,
            fp16_storage=fp16_storage,
        )
        self.lang = input_file.split("/")[-1].replace('.gz','').split('_')[0]
        self.part = input_file.split("/")[-1].replace('.gz','').split('_')[1]
        self.DSET_PREFIX = "{}_emds_{}".format(self.lang, '{}')
        self.curr_key = 0

        self.embedding_file_path = f"{cached_embeddings}{self.lang}_{self.part}.hdf5"
        self.embedding_file = h5py.File(self.embedding_file_path,"r")


    def name_output_file(self) -> str:
        shard_idx = extract_shard_id(self.input_file, default=self.input_file_idx)

        return os.path.abspath(
            os.path.join(
                self.output_dir,
                f"{self.outfile_prefix}.{shard_idx:03d}.{self.outfile_postfix}",
            )
        )

    def encode_to_np(
        self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]
    ) -> np.ndarray:
        embeddings = self.embedding_file[self.DSET_PREFIX.format(self.curr_key)]
        self.curr_key = self.curr_key + 1
        return np.stack( list(embeddings), axis=0 )

    def __exit__(self, _exc_type, _exc_value, _traceback):
        return None
