# SemEval 2025 Task 9: The Food Hazard Detection Challenge

## About
This repo contains solutions for [this](https://food-hazard-detection-semeval-2025.github.io/) project. In ST1 and ST2 the final folder contains the final submissions. To train you must use a distributed PyTorch run. For inference you can load the static model later and use the `submission_creator.py` script.


## Single run for Inference
```
python submission_creator.py
```

## Multi GPU run (4)
Use AWS instance with 4 GPUs such as `ml.g5.12xlarge`
```
bash
conda activate pytorch_p310
pip install transformers
cd SageMaker/
torchrun --nnodes=1 --nproc_per_node=4 training_torchrun.py 
```

## Multi GPU run (8)
Use AWS instance with 8 GPUs such as `ml.g6.48xlarge`
```
bash
conda activate pytorch_p310
pip install transformers
cd SageMaker/
torchrun --nnodes=1 --nproc_per_node=8 training_torchrun.py 
```

## Evaluation run (F1)
```
python f1.py
```