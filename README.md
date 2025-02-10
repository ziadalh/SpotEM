# SpotEM: Efficient Video Search for Episodic Memory

This repository contains a PyTorch implementation of our ICML-23 paper:

<p align="justify">
<a href="https://arxiv.org/pdf/2306.15850"> SpotEM: Efficient Video Search for Episodic Memory </a><br>
Santhosh Kumar Ramakrishnan<sup>1</sup> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Ziad Al-Halah<sup>2</sup> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp Kristen Grauman<sup>1,3</sup><br>
<i><sup>1</sup>The University of Texas at Austin &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp <sup>2</sup>University of Utah &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp <sup>3</sup>FAIR, Meta AI</i> <br>
Project website: <a href="https://vision.cs.utexas.edu/projects/spotem/">https://vision.cs.utexas.edu/projects/spotem/</a>
</p>


## Abstract

The goal in episodic memory (EM) is to search a long egocentric video to answer a natural language query (e.g., ``where did I leave my purse?"). Existing EM methods exhaustively extract expensive fixed-length clip features to look everywhere in the video for the answer, which is infeasible for long wearable camera videos that span hours or even days. We propose SpotEM, an approach to achieve efficiency for a given EM method while maintaining good accuracy. SpotEM consists of three key ideas: a novel clip selector that learns to identify promising video regions to search conditioned on the language query; a set of low-cost semantic indexing features that capture the context of rooms, objects, and interactions that suggest where to look; and distillation losses that address optimization issues arising from end-to-end joint training of the clip selector and EM model. Our experiments on 200+ hours of video from the Ego4D EM Natural Language Queries benchmark and three different EM models demonstrate the effectiveness of our approach: computing only 10%-25% of the clip features, we preserve 84%-95%+ of the original EM model's accuracy.

## Installation

* Install miniforge following instructions [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).
* Create new mamba / conda environment and activate it.
    ```
    mamba create -n spotem python=3.10
    mamba activate spotem
    ```
* Clone repository and define `SPOTEM_ROOT` variable.
    ```
    git clone ... <download_path>
    export SPOTEM_ROOT=<download_path>
    ```
* Install requirements.
    ```
    pip install -r requirements.txt
    ```
* Download NLTK resources
    ```
    python -c "import nltk; nltk.download('punkt_tab')"
    ```

# Data and model preparation
* Download Ego4D dataset from [here](https://ego4d-data.org/) and place it in `data/`.
* Download video features. Valid feature types: `clip`, `imagenet`, `room`, `interaction`, `object`, `egovlp`, `egovlp+imagenet`, `egovlp+rio`, `internvideo`, `internvideo+imagenet` and `internvideo+rio`.
    ```
    python utils/download_features.py --feature_types <feature_type_1> <feature_type_2> ...
    ```
* Run preprocessing script using:
    ```
    python VSLNet/utils/prepare_ego4d_dataset.py \
        --input_train_split data/nlq_train.json \
        --input_val_split data/nlq_val.json \
        --input_test_split data/nlq_test_unannotated.json \
        --output_save_path data/dataset/nlq_official_v1
    ```
* Download pretrained models for [EgoVLP](https://utexas.box.com/shared/static/m0ysg33slwq0geqm3nnset9ygwbsbrlc.zip), [InternVideo](https://utexas.box.com/shared/static/lj4fbf03s0lunxmjr9koppak1u7afgn7.zip), [ReLER](https://utexas.box.com/shared/static/0nxygni2bywyviediu9nk1x4vppyo3m7.zip) models. SpotEM's RIO feature encoders are also available: [room](https://utexas.box.com/shared/static/etetpo5emtqp5pi5n9rbecwgl3a3z4k7.tar), [interaction](https://utexas.box.com/shared/static/6xyqlcw6xmpfvsc6ya2jlngmv9xiaddt.tar) and [object](https://utexas.box.com/s/x29mgruwgzz6kfmlkqhckwjrolo9q42j) encoders.


# Training models
The training scripts for each model are provided in `scripts/training/<em_method_type>/`, where `<em_method_type>` can be `internvideo`, `reler` or `egovlp`. All training scripts expect `SPOTEM_ROOT` to be set as mentioned in the installation instructions. All scripts also require the GPU ids to be specified as the first argument. Some scripts optionally require the feature sampler efficiency to be specified as the second argument. Here is an example of training a random sampling baseline with the `internvideo` method, 4 GPUs and 50% sampling efficiency.
```
# Copy script to the required experiment path
cp $SPOTEM_ROOT/scripts/training/internvideo/random/train.sh <EXPERIMENT_DIR>
cd <EXPERIMENT_DIR>
bash train.sh "0,1,2,3" 0.5
```

Note that the models are evaluated on the validation split during the training process. For evaluating existing checkpoints, please refer to the [VSLNet](https://github.com/EGO4D/episodic-memory/tree/main/NLQ/VSLNet) and [RELER](https://github.com/NNNNAI/Ego4d_NLQ_2022_1st_Place_Solution) repositories for instructions.

# Citation
Please cite our work if you find this repository useful.
```
@inproceedings{ramakrishnan2023spotem,
  title={Spotem: Efficient video search for episodic memory},
  author={Ramakrishnan, Santhosh Kumar and Al-Halah, Ziad and Grauman, Kristen},
  booktitle={International Conference on Machine Learning},
  pages={28618--28636},
  year={2023},
  organization={PMLR}
}
```

# Acknowledgements
Our code is based on the following implementations of [VSLNet](https://github.com/EGO4D/episodic-memory/tree/main/NLQ/VSLNet) and [RELER](https://github.com/NNNNAI/Ego4d_NLQ_2022_1st_Place_Solution).
