import unicodedata

import torch


class VocabCompressor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mapping_tensor = None
        self.num_canonical = 0

    def build_mapping(self):
        vocab = self.tokenizer.get_vocab()
        normalized_to_id = {}
        raw_to_canonical = [-1] * len(vocab)

        for token_str, raw_id in vocab.items():
            text = self.tokenizer.decode([raw_id])
            text = unicodedata.normalize("NFKC", text)
            text = text.lower().strip()

            if not text:
                text = "<empty>"

            if text not in normalized_to_id:
                normalized_to_id[text] = len(normalized_to_id)

            raw_to_canonical[raw_id] = normalized_to_id[text]

        self.mapping_tensor = torch.tensor(raw_to_canonical, dtype=torch.long)
        self.num_canonical = len(normalized_to_id)

        print(f"Original Vocab Size: {len(vocab)}")
        print(f"Canonical Vocab Size: {self.num_canonical}")
        print(f"Reduction: {100 * (1 - self.num_canonical/len(vocab)):.2f}%")

        return self.mapping_tensor

if __name__=="__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    compressor = VocabCompressor(tokenizer)
    mapping = compressor.build_mapping()

    id1 = tokenizer.encode("Apple")[0]
    id2 = tokenizer.encode(" apple")[0]
    
    print(f"\nRaw ID for 'Apple': {id1}")
    print(f"Raw ID for ' apple': {id2}")
    
    canon1 = mapping[id1].item()
    canon2 = mapping[id2].item()
    
    print(f"Canonical ID for 'Apple': {canon1}")
    print(f"Canonical ID for ' apple': {canon2}")