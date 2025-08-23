# Train and save a new model
```python
python train/train_sac.py --episodes 100 --save-model ./models/my_sac_model.zip
```

# Load existing model and evaluate
```python
python train/train_sac.py --load-model ./models/my_sac_model.zip --episodes 0
```
# Custom training with different parameters
```python
python train/train_sac.py --episodes 50 --model-dir ./custom_models
