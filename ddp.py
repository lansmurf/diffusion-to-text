import torch
import torch.nn as nn
from transformers import T5EncoderModel
import keyword
import tokenize
from io import StringIO
import random
import numpy as np


def mask_tokens(code_snippet, language='python', mask_rate=0.15):
    masked_code = []
    if language == 'python':
        # Tokenize the Python code snippet
        tokens = list(tokenize.tokenize(StringIO(code_snippet).readline))
        for tok in tokens:
            # For keywords and identifiers (NAME tokens that aren't built-in functions), mask them based on mask_rate
            if (tok.type == tokenize.NAME and (
                    keyword.iskeyword(tok.string) or not tok.string.startswith('__'))) and random.random() < mask_rate:
                masked_code.append('<MASK>')
            else:
                masked_code.append(tok.string)
    elif language == 'bash':
        # This is a placeholder: you'll need a more sophisticated method for Bash, possibly using regex
        bash_keywords = ['if', 'else', 'fi', 'do', 'done', 'for', 'in', 'while', 'case', 'esac', 'echo', 'printf',
                         'export']
        for word in code_snippet.split():
            if word in bash_keywords and random.random() < mask_rate:
                masked_code.append('<MASK>')
            else:
                masked_code.append(word)
    return ' '.join(masked_code)


def add_gaussian_noise(embeddings, mean=0.0, std=0.1):
    noise = torch.randn_like(embeddings) * std + mean
    return embeddings + noise


import math
import torch.nn as nn


class Denoiser(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(Denoiser, self).__init__()
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.noise_prediction_layer = nn.Linear(d_model, d_model)

    def forward(self, noisy_embeddings, encoded_utterance, t, src_key_padding_mask=None):
        # Generate sinusoidal encoding for t
        t_embed = self.sinusoidal_encoding(t, noisy_embeddings.size(-1)).to(noisy_embeddings.device)
        # Add an extra dimension to make t_embed broadcastable to noisy_embeddings
        t_embed = t_embed.unsqueeze(1)

        # Add the timestep embedding to the noisy embeddings
        noisy_embeddings += t_embed

        # Self-attention layers
        for layer in self.self_attn_layers:
            noisy_embeddings = layer(noisy_embeddings, src_key_padding_mask=src_key_padding_mask)

        # Cross-attention layer
        attn_output, _ = self.cross_attn(noisy_embeddings, encoded_utterance, encoded_utterance)

        # Predict noise
        predicted_noise = self.noise_prediction_layer(attn_output)

        # Remove predicted noise to get denoised embeddings
        denoised_embeddings = attn_output - predicted_noise

        return denoised_embeddings, predicted_noise

    def sinusoidal_encoding(self, t, d_model):
        """
        Generates sinusoidal encodings for the timestep t.

        Parameters:
        - t: Current timestep, a scalar.
        - d_model: The dimension of the embeddings/model.

        Returns:
        - Sinusoidal encoding for t with shape [1, d_model].
        """
        position = torch.tensor([[t]], dtype=torch.float32)  # Shape: [1, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        sinusoidal_embedding = torch.zeros(1, d_model)
        sinusoidal_embedding[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_embedding[:, 1::2] = torch.cos(position * div_term)
        return sinusoidal_embedding


class DecoderWithCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, num_layers=1, dim_feedforward=2048, dropout=0.1):
        super(DecoderWithCrossAttention, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, denoised_embeddings, encoded_utterance):
        # Cross-attention in the decoder
        # Note: denoised_embeddings is the target (tgt) and encoded_utterance is the memory input for cross-attention
        output = self.transformer_decoder(denoised_embeddings, encoded_utterance)
        return output


class CodeEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CodeEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, code_tokens):
        # Convert code tokens to embeddings
        return self.embedding(code_tokens)


class T5EncoderBlock(nn.Module):
    def __init__(self, model_name='t5-medium'):
        super(T5EncoderBlock, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        # Process input tokens through the T5 encoder
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return encoder_outputs.last_hidden_state


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(input_dim, vocab_size)
        # Note: Softmax is not applied here as it's usually included in the loss function (e.g., nn.CrossEntropyLoss)

    def forward(self, input_embeddings):
        # Map embeddings to logits for each token in the vocabulary
        logits = self.linear(input_embeddings)
        return logits


class CODEFUSIONModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, denoiser, decoder, t5_model_name='t5-small'):
        super(CODEFUSIONModel, self).__init__()
        self.t5_encoder = T5EncoderBlock(t5_model_name)
        self.denoiser = denoiser  # Updated Denoiser
        self.decoder = decoder  # Updated Decoder
        self.classification_head = ClassificationHead(embedding_dim, vocab_size)

    def forward(self, embeddings, input_ids, timesteps=1200, attention_mask=None):
        # Encode natural language utterance
        encoded_utterance = self.t5_encoder(input_ids, attention_mask)

        for t in range(timesteps):
            gaussian_noise = torch.randn_like(embeddings) * square_root_noise_schedule(t)
            noisy_embeddings = embeddings + gaussian_noise

            # Pass noisy embeddings through denoiser
            denoised_embeddings, predicted_noise = model.denoiser(noisy_embeddings, encoded_utterance, t)

        # Process denoised embeddings through decoder with cross-attention to the encoded utterance
        decoded_embeddings = self.decoder(denoised_embeddings, encoded_utterance)

        # Generate logits for each code token position
        logits = self.classification_head(decoded_embeddings)
        return logits


def compute_loss(predicted_noise, actual_noise, denoised_embeddings, original_embeddings, logits, target_tokens):
    # Noise Prediction Loss
    noise_loss = torch.norm(predicted_noise - actual_noise, p=2)

    # Embedding Fidelity Loss
    embedding_loss = torch.norm(denoised_embeddings - original_embeddings, p=2)

    # Token Prediction Loss with Padding Ignored
    pad_token_id = t5_tokenizer.pad_token_id  # Get padding token ID
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    ce_loss = ce_loss_fn(logits.view(-1, logits.size(-1)), target_tokens.view(-1))

    # Combine losses
    total_loss = noise_loss + embedding_loss + ce_loss
    return total_loss


def square_root_noise_schedule(t, total_steps=1200, max_noise=1):
    """
    Calculates the noise level for a given step t using a square root schedule.

    Parameters:
    - t: Current diffusion step (0 <= t < total_steps).
    - total_steps: Total number of diffusion steps.
    - max_noise: Maximum noise level at the final step.

    Returns:
    - noise_level: Noise level at step t.
    """
    # Normalize the current step to a range between 0 and 1
    step_fraction = t / total_steps
    # Calculate the noise level using a square root schedule
    noise_level = max_noise * np.sqrt(step_fraction)
    return noise_level


from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel

class CodeDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(filepath, 'r', encoding='utf-8') as file:
            self.data = file.read().strip().split('\n\n')

        # Calculate the length of each code snippet before tokenization and truncation
        self.lengths = [len(snippet) for snippet in self.data]

        # Calculate weights based on the lengths, bias towards longer snippets
        self.weights = [len(snippet) ** 0.5 for snippet in
                        self.data]  # Square root to not overweight too much on longer snippets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code_snippet = self.data[idx]
        tokens = self.tokenizer.encode_plus(
            code_snippet,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)


def pre_sample_dataset(dataset, num_samples=60000):
    # Convert weights to probabilities
    total_weight = sum(dataset.weights)
    probabilities = [weight / total_weight for weight in dataset.weights]

    # Sample indices based on the calculated probabilities
    sampled_indices = np.random.choice(len(dataset), size=num_samples, replace=True, p=probabilities)

    # Create a subset of the original dataset based on the sampled indices
    sampled_dataset = Subset(dataset, indices=sampled_indices)
    return sampled_dataset

import os
import time
import time
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import memory_allocated, max_memory_allocated


def train(rank, world_size):
    # Setup the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    t5_tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')

    num_samples = 100000
    embedding_dim = 512
    dataset_path = 'PythonSnippets.txt'
    vocab_size = 32000
    batch_size = 64
    num_epochs = 1

    api_key = 'b40ad50fa130d6cc3b93e2c9a351630003a74dc6'

    if rank == 0:
        wandb.login(key=api_key)
        wandb.init(project='experiments')

    # Initialize models and move them to the current device
    embedding_layer = CodeEmbeddingLayer(vocab_size, embedding_dim).to(rank)
    denoiser = Denoiser(d_model=embedding_dim, nhead=8, num_layers=10).to(rank)
    decoder = DecoderWithCrossAttention(d_model=embedding_dim, nhead=8, num_layers=6).to(rank)

    # Wrap models with DDP
    embedding_layer = DDP(embedding_layer, device_ids=[rank])
    denoiser = DDP(denoiser, device_ids=[rank])
    decoder = DDP(decoder, device_ids=[rank])

    # Initialize the optimizer
    optimizer = optim.AdamW(
        list(embedding_layer.parameters()) +
        list(denoiser.parameters()) +
        list(decoder.parameters()), lr=5e-4, weight_decay=0
    )

    # Initialize dataset and dataloader
    original_dataset = CodeDataset(dataset_path, t5_tokenizer)
    sampled_dataset = pre_sample_dataset(original_dataset, num_samples=num_samples)
    sampler = DistributedSampler(sampled_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(sampled_dataset, batch_size=batch_size, sampler=sampler)

    scaler = GradScaler()  # Initialize the GradScaler for dynamic scaling

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Ensure shuffling at the start of each epoch
        loop = tqdm(dataloader, leave=True) if rank == 0 else dataloader  # Only show progress bar in the first process
        for batch_index, (input_ids, _) in enumerate(loop):
            start_time = time.time()
            optimizer.zero_grad()
            input_ids = input_ids.cuda(rank)

            with autocast():  # Enable automatic mixed precision
                embeddings = embedding_layer(input_ids)  # Original clean embeddings (L(y))

                # Randomly sample a starting timestep and define a range
                t = np.random.randint(0, 1200)
                # Add Gaussian noise based on the timestep
                gaussian_noise = torch.randn_like(embeddings) * square_root_noise_schedule(t)
                gaussian_noise_utterance = torch.randn_like(embeddings) * square_root_noise_schedule(t)
                noisy_embeddings = embeddings + gaussian_noise

                # Denoise the noisy embeddings
                denoised_embeddings, predicted_noise = denoiser(noisy_embeddings=noisy_embeddings,
                                                                encoded_utterance=gaussian_noise_utterance, t=t)

                # Calculate the noise prediction error
                noise_prediction_error = F.mse_loss(predicted_noise, gaussian_noise)

                # Decode the denoised embeddings to obtain the final decoded embeddings (D_s)
                decoded_embeddings = decoder(denoised_embeddings, encoded_utterance=gaussian_noise_utterance)

                # Calculate the embedding reconstruction error
                reconstruction_error = F.mse_loss(decoded_embeddings, embeddings)

                # Accumulate the losses for this timestep within the range
                timestep_loss = noise_prediction_error + reconstruction_error

                # Scale the accumulated loss from the sequential timesteps
                scaled_loss = scaler.scale(timestep_loss)
                scaled_loss.backward()
                scaler.step(optimizer)  # Optimizer step after processing the sequential timesteps
                scaler.update()

                total_loop_time = time.time() - start_time
                print(f"aaBatch {batch_index}: Total iteration time: {total_loop_time:.3f}s")

                if rank == 0:  # Only the first process logs the information
                    loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                    loop.set_postfix(loss=timestep_loss.item())
                    wandb.log({"loss": timestep_loss.item()})

    dist.destroy_process_group()  # Clean up

    # Save state dictionaries
    torch.save(embedding_layer.state_dict(), 'embedding_layer.pth')
    torch.save(denoiser.state_dict(), 'denoiser.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')

import torch.multiprocessing as mp

def main():
    world_size = torch.cuda.device_count()  # Number of available GPUs
    mp.spawn(train,
             args=(world_size,),  # Make sure this is a tuple
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
