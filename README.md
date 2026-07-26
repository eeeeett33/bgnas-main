## BGNAS

We investigate a new threat of implanting backdoors in GNN architectures that is independent of the model weights of GNNs. We propose BGNAS, a backdoored graph neural architecture search framework that uses graph neural architecture search algorithms to discover GNN architectures with backdoors. BGNAS jointly searches for backdoored GNN architectures and their associated trigger generators.

## Environment

The experiments require Python 3.8 or later and an NVIDIA GPU with a compatible CUDA installation is recommended.

Main dependencies:

- PyTorch
- PyTorch Geometric
- PyTorch Scatter
- NumPy
- SciPy
- scikit-learn
- nas-bench-graph

One possible installation is:

```bash
conda create -n bgnas python=3.10 -y
conda activate bgnas

# Install the PyTorch build compatible with your CUDA version first.
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter numpy scipy scikit-learn tqdm
```

Please follow the [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) installation guides if CUDA-specific wheels are required.

## Experiments

### 1. Neural Architecture Search

Run the architecture search with:

```bash
python darts_backdoor_minimal.py \
  --dataset Pubmed --device 0 --epochs 150 --seed 1781 \
  --hidden_dim 64 --layer_number 4 --dropout 0.5 \
  --model_lr 1e-3 --model_wd 5e-4 --arch_lr 1e-3 --arch_wd 1e-3 \
  --gen_lr 1e-3 --gen_wd 5e-4 --grad_clip 5.0 \
  --trigger_size 3 --target_class 0 --vs_size 40 --vs_number 40 \
  --selection_method cluster_degree --defense_mode prune --prune_thr 0 \
  --target_loss_weight 1.0 --homo_loss_weight 100.0 \
  --retrain_epochs 200 --retrain_lr 0.01 --retrain_wd 5e-4
```

Or run:

```bash
autogllight/examples/run.sh
```

### 2. Architecture Retraining

We provide high-performing architectures in `autogllight/examples/saved_models`. During retraining, the selected architecture is loaded, its model parameters are reinitialized, and the model is trained using a new random seed. The corresponding trigger generator is loaded together with the architecture and is frozen by default.

```bash
python retrain_fixed_gen.py \
  --dataset Pubmed --seed_load 1781 --seed_retrain 0 --device 0 \
  --trigger_size 3 --target_class 0 --vs_size 40 --vs_number 40 \
  --retrain_epochs 200 --train_lr 0.01 --weight_decay 5e-4 \
  --retrain_lr 0.01 --retrain_wd 5e-4 --trojan_epochs 200 \
  --selection_method cluster_degree --defense_mode none --prune_thr 0.1 \
  --target_loss_weight 1.0 --homo_loss_weight 100.0 --hidden 64
```

## Others

The code is provided solely for academic research purposes. Considering ethical impact and responsible disclosure, we have removed all released architectures and checkpoints except those presented in the paper. If you require access to these resources for legitimate research, please contact the corresponding author via email.
