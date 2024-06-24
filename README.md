# Conditional Transformer Language Model (CTRL)

This project implements an advanced version of the Conditional Transformer Language Model (CTRL) using the Hugging Face transformers library and datasets.

## Features

- Command-line argument parsing for easy configuration
- Logging for better tracking of training progress
- Model checkpointing to resume training
- Evaluation on a validation set
- Learning rate scheduling with warm-up
- Gradient clipping for stable training
- Text generation with temperature, top-k, and top-p sampling

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Datasets
- tqdm

Install the required packages using:

```
pip install torch transformers datasets tqdm
```
## Usage


1. Run the training script with desired arguments:
   ```
   python train_ctrl.py --dataset wikitext --dataset_config wikitext-2-raw-v1 --output_dir output --epochs 5 --batch_size 16 --learning_rate 3e-5
   ```

   This will train the CTRL model on the WikiText-2 dataset and save the trained model and checkpoints in the `output` directory.

2. To use the trained model for text generation, you can modify the `main()` function in the script or create a separate script:

   ```python
   from train_ctrl import CTRL, CTRLConfig, CTRLTokenizer, generate_text
   import torch

   # Load the tokenizer and config
   tokenizer = CTRLTokenizer.from_pretrained("ctrl")
   config = CTRLConfig(vocab_size=tokenizer.vocab_size)

   # Load the trained model
   model = CTRL(config)
   model.load_state_dict(torch.load("output/best_model.pth"))
   model.eval()

   # Generate text
   prompt = "Once upon a time"
   generated_text = generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95, device="cuda")
   print(generated_text)
   ```

## Customization

You can customize the model by modifying the following command-line arguments:

- `--dataset`: Name of the dataset to use (default: "wikitext")
- `--dataset_config`: Configuration of the dataset (default: "wikitext-2-raw-v1")
- `--output_dir`: Output directory for saving model and checkpoints (default: "output")
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 8)
- `--learning_rate`: Learning rate for the optimizer (default: 5e-5)
- `--warmup_steps`: Number of warmup steps for learning rate scheduler (default: 1000)
- `--max_grad_norm`: Maximum gradient norm for gradient clipping (default: 1.0)
- `--max_length`: Maximum sequence length for input texts (default: 128)

## License

This project is licensed under the MIT License.

## Acknowledgments

- Hugging Face for their excellent `transformers` and `datasets` libraries.
- The authors of the CTRL paper: Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, and Richard Socher.