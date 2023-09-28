# Learning Energy-based Model via Dual-MCMC Teaching ([Project Page](https://jcui1224.github.io/dual-MCMC-proj/))



## Train
Edit 'dataset = 'cifar10' / 'celeba64' in train.py for dataset. 
```python
CUDA_VISIBLE_DEVICES=gpu0 python train.py
```

### Test

```python
CUDA_VISIBLE_DEVICES=gpu0 python task_fid.py
```

Checkpoints for Cifar10 (avg 7.xx FID) and CelebA64 (avg 3.xx FID) are relaseed. Please update the checkpoint directory accordingly in task_fid.py.



