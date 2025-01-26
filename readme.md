# SemEval 2025 Task 9: The Food Hazard Detection Challenge

## Single GPU runs for Notebook
'conda create -n "python_3_12" python=3.12 pandas numpy matplotlib transformers scikit-learn'
'python -m ipykernel install --user --name python_3_12  --display-name "Python custom"'

## Multi GPU run (4)
bash
conda activate pytorch_p310
pip install transformers
cd SageMaker/
torchrun --nnodes=1 --nproc_per_node=4 test.py 
test 2 is experimental

## Multi GPU run (8)
bash
conda activate pytorch_p310
pip install transformers
cd SageMaker/
torchrun --nnodes=1 --nproc_per_node=8 test.py 
test 2 is experimental