## Overview
This is a joint work by [Yiren Jian](https://cs.dartmouth.edu/~yirenjian/), [Chongyang Gao](https://gcyzsl.github.io/), [Yunjie Zhao](https://scholar.google.com/citations?user=nXWIMFcAAAAJ&hl=en), [Chen Zeng](https://physics.columbian.gwu.edu/chen-zeng) and [Soroush Vosoughi](https://www.cs.dartmouth.edu/~soroush/). The paper is under review.

## Requirements

In this repo, we provide the script and model for running inference (testing). Any machines with a CPU and an Ubuntu system should work. The GPU is not required for inference. Assuming you have [Anaconda](https://www.anaconda.com/) installed, or you can do
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
```

After that, you will need two major software packages: `pytorch` and `pydca`. The following commands will create a virtual environment and install the necessary packages.

```bash
conda create -n pytorch-1.8 python=3.7
conda activate pytorch-1.8
pip install tqdm
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pydca
pip install tensorboard
```

## Usage

The input is a RNA MSA (see [examples](RNA_TESTSET/MSA_pydca)) and the output is predicted contact scores.
```bash
python run_inference.py --input_MSA RNA_TESTSET/MSA_pydca/RF00001.faclean
```

The outputs are saved as `outputs/dist.txt` and `outputs/pred.txt`.

## Training
The training scripts will be released soon.
- [x] `preprocess_msa.py`
- [ ] `preprocess_feat.py`
- [ ] `train_val_transfer.py`
- [ ] `eval_transfer.py`

You will first need to translate RNA MSA from nucleotide to amino acids. For example, from `AUCG` to `HETL`.
```bash
python preprocess_msa.py --AminoAicds "HETL"
```

Next, `preprocess_feat.py` will pre-extract CoT features from different layers (inputs to the transfer model) and ground truth RNA contacts from PDBs (as ground truth labels).
```bash
python preprocess_feat.py
```

The following script will train the model on the training set, pick the best model based on validation, and finally evaluate on the testing set.
```bash
for MIN_LR in 0.0
do
    for EPOCH in 100 300 500
    do
        for BATCH_SIZE in 4 8 12 16
        do
            CUDA_VISIBLE_DEVICES=0,1,2,3 python train_val_transfer.py --scheduler_type 'CosineLR' --min_lr $MIN_LR --batch_size $BATCH_SIZE --total_epoch $EPOCH --feature_list 0 1 2 3 4 5 6
        done
    done
done
```
The trained models are saved in `saved_models/` and the training logs are stored in `tensorboard_dir/`.  Check [this](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server) for how to visualize training/testing loss/curve on your local machine.

## License
Our work is built on two prior works [coevolution_transformer](https://github.com/microsoft/ProteinFolding/tree/main/coevolution_transformer) and [coconet](https://github.com/KIT-MBS/coconet), both are MIT licensed.
