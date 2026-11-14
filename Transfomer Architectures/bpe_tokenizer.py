from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import trainers, models
import os
from tokenizers import Tokenizer



def load_train_bpe_forward(
    vocab_size: int,
    save_prefix: str,
    save_name: str = "bpe_joint.json",
    src_corpus_file: str = "trainsrc.txt",
    tgt_corpus_file: str = "traintgt.txt",
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    save_path = f"{save_prefix}_{save_name}"

    if os.path.exists(save_path):
        print(f"loading the tokenizer from {save_path}")
        return Tokenizer.from_file(save_path)

    for corpus_file in (src_corpus_file, tgt_corpus_file):
        if not os.path.isfile(corpus_file):
            raise FileNotFoundError(
                f"Corpus file not found: {corpus_file}. "
                "Run dataset.py first or pass the generated corpus paths "
                "with src_corpus_file and tgt_corpus_file."
            )

    if special_tokens is None:
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.normalizer = NFKC()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    tokenizer.train([src_corpus_file, tgt_corpus_file], trainer)
    tokenizer.save(save_path)
    print(f"saved tokenizer to {save_path}")
    return tokenizer


if __name__ == "__main__":
    tokenizer = load_train_bpe_forward(vocab_size=10, save_prefix="bpe")
    print(f"tokenizer vocab size: {tokenizer.get_vocab_size()}")
    print("Pad token id:", tokenizer.token_to_id("[PAD]"))
    print("CLS token id:", tokenizer.token_to_id("[CLS]"))
    print("SEP token id:", tokenizer.token_to_id("[SEP]"))