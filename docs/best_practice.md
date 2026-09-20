## Best Practices
  - 1. Use argparse to take in command line parameters into your scripts (e.g. python train.py --lr 0.01)
  - 2. Use tqdm to show training progress
  - 3. Write custom training loops rather than model.fit from tensorflow.
  - 4. Implement checkpointing and best so-far mechanism to save training progress and keep track of best models.
  - 5. Understand the loss function and evaluation metrics you use.
  - 6. Test your code on a component basis to ensure each part is working as expected.