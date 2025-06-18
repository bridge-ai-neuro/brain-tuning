# Improving Semantic Understanding in Speech Language Models via Brain-tuning


[Paper ICLR Page](https://iclr.cc/virtual/2025/poster/30063) | [arXiv](https://arxiv.org/abs/2410.09230) | [Poster](https://iclr.cc/virtual/2025/poster/30063)

Code for the paper: 

Omer Moussa, Dietrich Klakow, and Mariya Toneva. Improving Semantic Understanding in Speech Language Models via Brain-tuning. In _International Conference on Learning Representations (ICLR)_, 2025. 

## Abstract
Speech language models align with human brain responses to natural language to an impressive degree. However, current models rely heavily on low-level speech features, indicating they lack brain-relevant semantics which limits their utility as model organisms of semantic processing in the brain. In this work, we address this limitation by inducing brain-relevant bias directly into the models via fine-tuning with fMRI recordings of people listening to natural stories--a process we name brain-tuning. After testing it on 3 different pretrained model families, we show that brain-tuning not only improves overall alignment with new brain recordings in semantic language regions, but also reduces the reliance on low-level speech features for this alignment. Excitingly, we further show that brain-tuning leads to 1) consistent improvements in performance on semantic downstream tasks and 2) a representational space with increased semantic preference. Our results provide converging evidence, for the first time, that incorporating brain signals into the training of language models improves the models’ semantic understanding.

## Datasets and Preprocessing

 We use the Full Moth Radio Hour [Dataset](https://www.nature.com/articles/s41597-023-02437-z). It can be downloaded from [here](https://openneuro.org/datasets/ds003020). No further preprocessing steps for the fMRI responses are needed because the `derivatives/preprocessed_data` in the dataset is already processed as per the original paper recommendation. For the stimuli, the only preprocessing needed is to make sure that the sampling rate is 16000. 

 To easily use the TextGrids and the TRFiles, we recommend downloading them from [Antonello et. al, 2024](https://utexas.app.box.com/v/EncodingModelScalingLaws). 

 We use Noise Ceiling to filter noisy voxels during brain-tuning and brain alignment calculations. They can be computed and saved from `spe_and_cc_norm` in `eval_utils.py`, or alternatively loaded directly from the provided `subject_NCs` folder. 

## How to Brain-tune models

After downloading the fMRI dataset and its metadata (the Grids and TRFiles) and installing the requirements (from requirements.txt), brain-tuning can be carried out using `brain_trainer_story.py` as follows: 

`python brain_trainer_story.py --model_name {huggingface pretrained model path} --subject {fMRI subject number} --num_epochs {training epochs}`. There are many more arguments that control logging and training that could be found in the code. 

To compute brain alignment for the pretrained or the brain-tuned models, use `brain_evaluate.py` as follows:

`python brain_evaluate.py --model_name {huggingface pretrained model path} --subject {fMRI subject number} --logs_dir {directory containing saved weights of the model}`

Downstream Evaluation can be carried out similarly; for instance, to test the model on phonemes prediction run:
`python eval_phonemes.py --model_name {huggingface pretrained model path} --model_path {path to the saved .pth model weights}`. Many more arguments that control logging and training can be found in the code.



## How to Cite
To cite our work, we recommend using the following BibTeX: 
```
@inproceedings{
moussa2024improvings,
title={Improving Semantic Understanding in Speech Language Models via Brain-tuning},
author={Omer Moussa, Dietrich Klakow, and Mariya Toneva},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=KL8Sm4xRn7}
}
```


# Contributors
[Omer Moussa](https://www.mpi-sws.org/people/omoussa/) (omoussa@mpi-sws.org) - Corresponding Author

[Prof. Dietrich Klakow](https://www.lsv.uni-saarland.de/people/dietrich-klakow/) (dietrich.klakow@lsv.uni-saarland.de)

[Prof. Mariya Toneva](https://mtoneva.com/) (mtoneva@mpi-sws.org)

