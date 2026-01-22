import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from config import EngramConfig
from engram import EngramLayer


class GPT2WithEngram(PreTrainedModel):
    config_class = EngramConfig

    def __init__(self, config, vocab_map):
        super().__init__(config)
        self.config = config
        self.gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

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

if __name__=="__main__":
    from transformers import AutoTokenizer

    from compression import VocabCompressor
    from config import EngramConfig

    config = EngramConfig()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    compressor = VocabCompressor(tokenizer)
    mapping = compressor.build_mapping()

    model = GPT2WithEngram(config, mapping)
    
    print(model)
    print(model(torch.tensor([[1, 10]])))