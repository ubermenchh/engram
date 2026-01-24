import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from config import EngramConfig
from engram import EngramLayer


class GPT2WithEngram(PreTrainedModel):
    config_class = EngramConfig

    def __init__(self, config, vocab_map, nanogpt_config):
        super().__init__(config)
        self.config = config

        if nanogpt_config is None:
            from transformers import GPT2Config
            nanogpt_config = GPT2Config(
                vocab_size=config.vocab_size,
                n_embd=config.d_model,
                n_layer=6,
                n_head=6,
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1
            )
        self.gpt2 = AutoModelForCausalLM.from_config(nanogpt_config)

        if vocab_map is None:
            vocab_map = torch.zeros(config.vocab_size, dtype=torch.long)

        self.engram = EngramLayer(config, vocab_map)
        self.target_layer_idx = config.injection_layer

    def forward(self, input_ids, labels=None, attention_mask=None):
        self.current_input_ids = input_ids

        def engram_hook(module, args, output):
            hidden_state = output[0]
            engram_out = self.engram(self.current_input_ids, hidden_state)
            # print(f"engram_out: {engram_out.abs().mean()}")
            new_hidden = hidden_state + engram_out
            return (new_hidden,) + output[1:]

        layer_module = self.gpt2.transformer.h[self.target_layer_idx]
        hook_handle = layer_module.register_forward_hook(engram_hook)

        try:
            outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        finally:
            hook_handle.remove()
            self.current_input_ids = None

        return outputs

AutoModelForCausalLM.register(EngramConfig, GPT2WithEngram)

if __name__=="__main__":
    from transformers import AutoTokenizer, GPT2Config

    from compression import VocabCompressor
    from config import EngramConfig

    config = EngramConfig()
    nanogpt_config = GPT2Config(
        n_embd=384,
        n_layer=6,
        n_head=6
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    compressor = VocabCompressor(tokenizer)
    mapping = compressor.build_mapping()

    model = GPT2WithEngram(config, mapping, nanogpt_config)
    
    print(model)
    print("NanoGPT Params:", sum(p.numel() for p in model.parameters()))
    print(model(torch.tensor([[1, 10]])))