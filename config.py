from transformers import AutoConfig, PretrainedConfig


class EngramConfig(PretrainedConfig):
    model_type = "engram"

    def __init__(
        self,
        vocab_size=50257,
        d_model=384,
        ngram_orders=[2, 3],
        num_heads=2,
        bucket_size=100_000,
        engram_dim=192,
        injection_layer=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ngram_orders = ngram_orders
        self.num_heads = num_heads
        self.bucket_size = bucket_size
        self.engram_dim = engram_dim
        self.injection_layer = injection_layer
        super().__init__(**kwargs)

AutoConfig.register("engram", EngramConfig)