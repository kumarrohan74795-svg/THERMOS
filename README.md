# THERMOS: Thermally-Aware Multi-Objective Scheduling for Heterogeneous Multi-Chiplet PIM Architectures
This repository contains the code for the paper "THERMOS: Thermally-Aware Multi-Objective Scheduling for Heterogeneous Multi-Chiplet PIM Architectures" presented at ESWEEK 2025.

To install dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To train the model, run the following command:

```bash
python train_THERMOS_mapper_parallel.py
```
Training script will automatically save the model checkpoints in the `eval_models` directory.
This script uses Multi-Processing to train the model in parallel. The actor model is Differential Decision Tree (DDT) and the critic model is a Multi-Layer Perceptron (MLP).

## Evaluation
To evaluate the model, run the following command:

```bash
python test_THERMOS_mapper.py
```
This script will load the model checkpoint from the `final_model` directory. The results will be saved in the `rl_results` directory.

One trained model checkpoint is provided in the `final_model` directory. This checkpoint was used to generate the results in the paper. 

## Reference

If you use this code in your research, please cite our paper:
```bibtex
@article{rohan2026thermos,
  title={THERMOS: Thermally-Aware Multi-Objective Scheduling of AI Workloads on Heterogeneous Multi-Chiplet PIM Architectures},
  author=kumar, Alish and Pfromm, Lukas and Sharma, Harsh and Doppa, Jana and Pande, Partha and Ogras, Umit},
  journal={ACM Transactions on Embedded Computing Systems},
  year={2026}
}
