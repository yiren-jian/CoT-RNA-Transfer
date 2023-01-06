## Overview
This is a joint work by [Yiren Jian](https://cs.dartmouth.edu/~yirenjian/), [Chongyang Gao](https://gcyzsl.github.io/), [Yunjie Zhao](https://scholar.google.com/citations?user=nXWIMFcAAAAJ&hl=en), [Chen Zeng](https://physics.columbian.gwu.edu/chen-zeng) and [Soroush Vosoughi](https://www.cs.dartmouth.edu/~soroush/). The paper is under review.

## Requirements

In this repo, we provide the script and model for running inference (testing). Any machines with a CPU and an Ubuntu system should work. The GPU is not required. Assuming you have [Anaconda](https://www.anaconda.com/) installed, you will need two major software packages: `pytorch` and `pydca`. The following commands will create a virtual environment and install the necessary packages.

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

The outputs are saved in `outputs/dist.txt` and `outputs/pred.txt`.

## Training
The training scripts will be released soon.

## License
Our work is built on two prior works [coevolution_transformer](https://github.com/microsoft/ProteinFolding/tree/main/coevolution_transformer) and [CoCoNet](https://github.com/KIT-MBS/coconet), which are both MIT Licensed.
