# Conditional Transformer Language Model (CTRL)

This project implements a Conditional Transformer Language Model (CTRL) using the Hugging Face transformers library and datasets.

## Overview

The CTRL model is a powerful language model that can generate text based on various control codes. This implementation uses the Hugging Face `transformers` library for the base CTRL architecture and the `datasets` library to load the WikiText-2 dataset for training.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Datasets

You can install the required packages using:

```
pip install torch transformers datasets
```

## Usage

1. Run the training script:
   ```
   python train_ctrl.py
   ```

   This will train the CTRL model on the WikiText-2 dataset and save the trained model as `ctrl_model.pth`.

2. To use the trained model for text generation, you can load it and use it as follows:

   ```python
   import torch
   from transformers import CTRLTokenizer
   from model import CTRL, CTRLConfig

   # Load the tokenizer and config
   tokenizer = CTRLTokenizer.from_pretrained("ctrl")
   config = CTRLConfig(vocab_size=tokenizer.vocab_size)

   # Load the trained model
   model = CTRL(config)
   model.load_state_dict(torch.load("ctrl_model.pth"))
   model.eval()

   # Generate text
   input_text = "Once upon a time"
   input_ids = tokenizer.encode(input_text, return_tensors="pt")
   output = model.generate(input_ids, max_length=100, num_return_sequences=1)
   generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
   print(generated_text)
   ```

## Customization

You can customize the model by modifying the following parameters in the `train_ctrl` function:

- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `lr`: Learning rate for the optimizer

You can also experiment with different datasets by changing the dataset loading line:

```python
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Hugging Face for their excellent `transformers` and `datasets` libraries.
- The authors of the CTRL paper: Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, and Richard Socher.
