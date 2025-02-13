# SemEval 2025 Task 9: The Food Hazard Detection Challenge

## About
This repo contains solutions for [this](https://food-hazard-detection-semeval-2025.github.io/) project. In ST1 and ST2 the final folder contains the final submissions. To train you must use a distributed PyTorch run. For inference you can load the static model later and use the `submission_creator.py` script.

## Learn about the Network
Read the `SemEval 2025 Task 9` notebook

## Run the code

### Multi GPU training (8)
Navigate to `ST1/final` or `ST2/final`\
Use a Sage Maker AWS instance with 8 GPUs such as `ml.g6.48xlarge` and open the provided terminal.\
Run:
```
bash
conda activate pytorch_p310
pip install transformers
cd SageMaker/
torchrun --nnodes=1 --nproc_per_node=8 training_torchrun.py 
```

### Single run for Inference
Navigate to `ST1/final` or `ST2/final`
```
python submission_creator.py
```

### Evaluation run (F1)
Navigate to `ST1/final` or `ST2/final`
```
python f1.py
```