import io
import re
import torch
from torchtext.vocab import Vocab  # Import Vocab
from collections import OrderedDict
import torch.nn as nn
import os
from collections import Counter
from fastapi import HTTPException  # Added import for HTTPException
import logging

# Define the RNN class (same as in training)
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        _, (hidden, _) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Load model
def load_model(model_path: str, device: torch.device, vocab_size: int, embed_dim: int, rnn_hidden_size: int, fc_hidden_size: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.eval()

# Tokenizer function
def tokenizer(text: str):
    text = re.sub('<[^>]*>', '', text)  # Remove HTML tags
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub(r'[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', '')
    return text.split()

# Load vocabulary
def load_vocab(vocab_path):
    try:
        vocab = torch.load(vocab_path)
        if hasattr(vocab, 'get_stoi'):
            vocab_dict = vocab.get_stoi()
        elif hasattr(vocab, 'stoi'):
            vocab_dict = vocab.stoi
        else:
            raise AttributeError(f"'{type(vocab).__name__}' object has no 'stoi' attribute.")
        return vocab
    except Exception as e:
        raise RuntimeError(f"Failed to load vocabulary: {str(e)}")

# Inference function
def predict(model, vocab, text, device):
    try:
        # Tokenize the text and convert it to tensor
        tokens = tokenizer(text)
        text_tensor = torch.tensor([vocab[token] for token in tokens], dtype=torch.int64).unsqueeze(0)

        # Move tensors to the same device as the model
        text_tensor = text_tensor.to(device)
        lengths = torch.tensor([text_tensor.size(1)], dtype=torch.int64).to(device)

        # Ensure the model is on the correct device
        model = model.to(device)

        # Run the model to get the output (prediction)
        output = model(text_tensor, lengths)

        # Convert the raw output to a probability (if applicable) and assign a label
        prediction = output.item()  # Output already passes through sigmoid in the model
        label = "positive" if prediction > 0.5 else "negative"

        # Return the model's prediction as a dictionary
        return {"prediction": label, "score": prediction}

    except Exception as e:
        # Log the error and return a more helpful message
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

