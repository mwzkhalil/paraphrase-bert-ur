# Urdu Paraphrase Model

This repository contains code for training a paraphrase model on a Urdu CSV dataset using BERT as the base model.

## Requirements

- torch==1.9.0
- torchvision==0.10.0
- pandas==1.3.0
- scikit-learn==0.24.2
- transformers==4.11.3

## Dataset

The Urdu CSV dataset used for training the model should be placed in the root directory with the name `urdu_dataset.csv`. Ensure that the dataset has two columns: 'text' and 'paraphrase', representing the original text and its corresponding paraphrase.

## Usage

1. Install the required dependencies by running the following command:

```shell
pip install -r requirements.txt
```

2.Run the `train.py` script to train the paraphrase model:

3. After training, the trained model will be saved as `paraphrase_model.pt`.

4. Use the trained model for inference or evaluation in your own applications.

## Model Architecture

The paraphrase model uses BERT as the base model with a linear layer on top. The model architecture is defined in the `model.py` file. Feel free to customize the model architecture based on your specific requirements.

## License

This project is licensed under the [MIT License](LICENSE).
