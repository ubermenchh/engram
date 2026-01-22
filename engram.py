import torch
import torch.nn as nn
import torch.nn.functional as F


class EngramLayer(nn.Module):
    vocab_map: torch.Tensor
    hash_weights: torch.Tensor

    def __init__(self, config, mapping_tensor):
        super().__init__()
        self.config = config
        self.register_buffer("vocab_map", mapping_tensor)
        self.num_lookups = len(config.ngram_orders) * config.num_heads
        self.head_dim = config.engram_dim // self.num_lookups
        self.tables = nn.ModuleList([
            nn.Embedding(config.bucket_size, self.head_dim)
            for _ in range(self.num_lookups)
        ])

        max_n = max(config.ngram_orders)
        self.register_buffer("hash_weights", torch.randint(1, 100000, (self.num_lookups, max_n)))

        self.gate_proj = nn.Linear(config.engram_dim, config.d_model, bias=False)
        self.value_proj = nn.Linear(config.engram_dim, config.d_model, bias=False)
        self.norm = nn.RMSNorm(config.d_model)
        self.conv = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=4,
            padding=3,
            groups=config.d_model
        )

    def get_ngrams(self, input_ids):
        B, L = input_ids.shape
        canonical_ids = self.vocab_map[input_ids]
        retrieved_parts = []
        table_idx = 0

        for n in self.config.ngram_orders:
            for k in range(self.config.num_heads):
                padded = F.pad(canonical_ids, (n - 1, 0), value=0)
                windows = padded.unfold(1, n, 1)
                weights = self.hash_weights[table_idx, :n]
                hashed_vals = (windows.float() @ weights.float()).long()
                indices = hashed_vals % self.config.bucket_size

                vector_part = self.tables[table_idx](indices)
                retrieved_parts.append(vector_part)

                table_idx += 1

        return torch.cat(retrieved_parts, dim=-1)

    def forward(self, input_ids, hidden_state):
        memory_raw = self.get_ngrams(input_ids)
        
        key = self.norm(self.gate_proj(memory_raw))
        query = self.norm(hidden_state)

        score = (query * key).sum(dim=-1, keepdim=True)
        gate_val = torch.sigmoid(score)

        value = self.value_proj(memory_raw)
        gated_value = gate_val * value

        conv_input = gated_value.permute(0, 2, 1)
        conv_out = self.conv(conv_input)
        conv_out = conv_out[:, :, :hidden_state.size(1)]
        conv_out = conv_out.permute(0, 2, 1)
    
        output = F.silu(conv_out) + gated_value
        return output


if __name__=="__main__":
    from transformers import AutoTokenizer

    from compression import VocabCompressor
    from config import EngramConfig

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = EngramConfig()
    compressor = VocabCompressor(tokenizer)
    mapping = compressor.build_mapping()
    engram_layer = EngramLayer(config, mapping)

    input_ids = torch.randint(0, 1000, (1, 10))
    hidden_state = torch.randn(1, 10, 768)

    out = engram_layer(input_ids, hidden_state)

    print(engram_layer)
    print("Output shape: ", out.shape)