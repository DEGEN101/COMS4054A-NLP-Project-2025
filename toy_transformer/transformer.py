import torch as T
import torch.nn as nn
import numpy as np

from .utilities import get_attentions, split_heads, combine_heads, generate_mask


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_size: int, n_heads: int) -> None:
        super().__init__()

        if embedding_size % n_heads != 0:
            raise Exception("[!] Embedding size not divisible by number of attention heads.")

        self.embedding_size = embedding_size
        self.n_heads = n_heads

        # Size of representation vector
        self.d_k = embedding_size // n_heads

        # Query weights
        self.W_Q = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        # Key weights
        self.W_K = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        # Value weights
        self.W_V = nn.Linear(in_features=embedding_size, out_features=embedding_size)

        # Output layer
        self.W_O = nn.Linear(in_features=embedding_size, out_features=embedding_size)

    def forward(self, Q: T.Tensor, K: T.Tensor, V: T.Tensor, mask: T.Tensor | None = None) -> T.Tensor:
        queries = split_heads(self.W_Q(Q), self.n_heads, self.d_k).transpose(1, 2)
        keys = split_heads(self.W_K(K), self.n_heads, self.d_k).transpose(1, 2)
        values = split_heads(self.W_V(V), self.n_heads, self.d_k).transpose(1, 2)

        attentions, self.attention_scores = get_attentions(queries, keys, values, self.d_k, mask)
        return self.W_O(combine_heads(attentions.transpose(1, 2)))


class FeedForwardBlock(nn.Module):
    def __init__(self, embedding_size: int, dim: int, dropout_prob: float) -> None:
        super().__init__()

        self.linear_layer1 = nn.Linear(in_features=embedding_size, out_features=dim)
        self.linear_layer2 = nn.Linear(in_features=dim, out_features=embedding_size)
        self.dropout_layer = nn.Dropout(dropout_prob)

    def forward(self, x):
        z = T.relu(self.linear_layer1(x))
        z = self.dropout_layer(z)
        return self.linear_layer2(z)


class PositionalEncodingBlock(nn.Module):
    def __init__(self, embedding_size: int, max_sequence_length: int) -> None:
        super().__init__()

        positional_encoding = T.zeros([max_sequence_length, embedding_size])
        position = T.arange(0, max_sequence_length, dtype=T.float).reshape(-1, 1)
        denominator = T.exp(T.arange(0, embedding_size, 2).to(T.float) * -(np.log(10000.0) / embedding_size))

        positional_encoding[:, 0::2] = T.sin(position / denominator)
        positional_encoding[:, 1::2] = T.cos(position / denominator)

        self.register_buffer("positional_encoding", positional_encoding.unsqueeze(0))

    def forward(self, x: T.Tensor):
        seq_len = x.size(1)
        return x + self.positional_encoding[:, :seq_len].to(x.device)


class EncoderBlock(nn.Module):
    def __init__(self, embedding_size: int, n_attention_heads: int, ff_dim:int, dropout_prob: float) -> None:
        super().__init__()

        self.mha_block = MultiHeadAttentionBlock(embedding_size, n_attention_heads)
        self.ff_block = FeedForwardBlock(embedding_size, ff_dim, dropout_prob)

        self.normalization_layer1 = nn.LayerNorm(embedding_size)
        self.normalization_layer2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x: T.Tensor):
        # Self attention
        mha_output = self.mha_block(x, x, x)
        z = self.normalization_layer1(x + self.dropout(mha_output))

        ff_output = self.ff_block(z)

        return self.normalization_layer2(z + self.dropout(ff_output))


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size: int, n_attention_heads: int, ff_dim:int, dropout_prob: float) -> None:
        super().__init__()

        self.masked_mha_block = MultiHeadAttentionBlock(embedding_size, n_attention_heads)
        self.mha_block = MultiHeadAttentionBlock(embedding_size, n_attention_heads)
        self.ff_block = FeedForwardBlock(embedding_size, ff_dim, dropout_prob)

        self.normalization_layer1 = nn.LayerNorm(embedding_size)
        self.normalization_layer2 = nn.LayerNorm(embedding_size)
        self.normalization_layer3 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: T.Tensor, encoder_output: T.Tensor, mask: T.Tensor):
        # Self attention
        masked_mha_output = self.masked_mha_block(x, x, x, mask)
        z = self.normalization_layer1(x + self.dropout(masked_mha_output))

        # Cross attention
        mha_output = self.mha_block(z, encoder_output, encoder_output)
        z = self.normalization_layer2(z + self.dropout(mha_output))

        ff_output = self.ff_block(z)

        return self.normalization_layer3(z + self.dropout(ff_output))


class Transformer(nn.Module):
    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_size: int, 
                n_attention_heads: int, n_blocks: int, max_sequence_length: int, ff_dims:int, 
                dropout_prob: float, device: str | T.device = "cpu") -> None:
        super().__init__()

        self.device = T.device(device)

        self.encoder_embedding = nn.Embedding(num_embeddings=source_vocab_size, embedding_dim=embedding_size).to(self.device)
        self.decoder_embedding = nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=embedding_size).to(self.device)
        self.pe_block = PositionalEncodingBlock(embedding_size, max_sequence_length).to(self.device)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embedding_size, n_attention_heads, ff_dims, dropout_prob).to(self.device) for _ in range(n_blocks)
        ])

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embedding_size, n_attention_heads, ff_dims, dropout_prob).to(self.device) for _ in range(n_blocks)
        ])

        self.linear_layer = nn.Linear(in_features=embedding_size, out_features=target_vocab_size).to(self.device)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, source_sequence, target_sequence):
        source_sequence = source_sequence.to(self.device)
        target_sequence = target_sequence.to(self.device)

        mask = generate_mask(target_sequence).to(self.device)

        source_embedded = self.dropout(self.pe_block(self.encoder_embedding(source_sequence)))
        target_embedded = self.dropout(self.pe_block(self.decoder_embedding(target_sequence)))

        encoder_output = source_embedded
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output)

        decoder_output = target_embedded
        for block in self.decoder_blocks:
            decoder_output = block(decoder_output, encoder_output, mask)

        return self.linear_layer(decoder_output)
