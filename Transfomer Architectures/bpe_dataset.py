import os

import rich
import rich
import torch
from torch.utils.data import Dataset

def get_seed(verbose, seed = 42):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if verbose:
        print("using this device", device)
    elif torch.backends.mps.is_available():
        values = torch.values("mps")
    if verbose:
        print("using slicion gpu", values)
    else:
        torch.device("cpu")
    if verbose:
        print("using cpu")

def create_padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    # True=allowed, False=masked
    return ids != pad_id




class TranslationDataset(Dataset):
    def __init__(self, file_prefix, tokenizer, max_seq_len=128, is_debug=True):
        super(TranslationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.debug = is_debug

        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.sos_id = self.tokenizer.token_to_id("[CLS]")
        self.eos_id = self.tokenizer.token_to_id("[SEP]")
        if self.pad_id is None or self.sos_id is None or self.eos_id is None:
            raise ValueError("Tokenizer must define [PAD], [CLS], and [SEP] tokens")

        self.source_texts, self.target_texts = self.load_id_pairs(file_prefix)

    def load_id_pairs(self, file_prefix):
        src_path = file_prefix + ".src_ids.pt"
        tgt_path = file_prefix + ".tgt_ids.pt"

        assert os.path.exists(src_path), f"Source ID file not found: {src_path}"
        assert os.path.exists(tgt_path), f"Target ID file not found: {tgt_path}"

        source_ids = torch.load(src_path)
        target_ids = torch.load(tgt_path)

        assert len(source_ids) == len(target_ids), (
            "Mismatched number of sequences in source and target ID files"
        )

        if self.debug:
            source_ids = source_ids[:100]
            target_ids = target_ids[:100]
        return source_ids, target_ids

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        return {
            "src_ids": self.source_texts[idx][: self.max_seq_len],
            "tgt_ids": self.target_texts[idx][: self.max_seq_len],
        }

    @staticmethod
    def translation_collate(
        batch, pad_id: int, sos_id: int, eos_id: int, max_len: int | None = None
    ):
        src_list = [item["src_ids"] for item in batch]
        tgt_list = [item["tgt_ids"] for item in batch]

        if max_len is not None:
            src_list = [x[:max_len] for x in src_list]
            tgt_list = [x[:max_len] for x in tgt_list]

        decoder_list = [t[:-1] for t in tgt_list]
        labels_list = [t[1:] for t in tgt_list]

        src_max = max(x.numel() for x in src_list)
        dec_max = max(x.numel() for x in decoder_list)

        def pad_1d(x: torch.Tensor, length: int) -> torch.Tensor:
            x = x.to(torch.long)
            if x.numel() == length:
                return x
            return torch.cat(
                [x, x.new_full((length - x.numel(),), pad_id, dtype=torch.long)]
            )

        encoder_input_ids = torch.stack(
            [pad_1d(x, src_max) for x in src_list], dim=0
        )
        decoder_input_ids = torch.stack(
            [pad_1d(x, dec_max) for x in decoder_list], dim=0
        )
        labels = torch.stack([pad_1d(x, dec_max) for x in labels_list], dim=0)

        return {
            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "encoder_mask": create_padding_mask(encoder_input_ids, pad_id),
            "decoder_mask": create_padding_mask(decoder_input_ids, pad_id),
        }
        
        
        
@torch.no_grad
def sample(model, tokenizer, src: str = ""):
    model.eval()
    if src == "":
        english = "Several years ago here at TED, Peter Skillman  introduced a design challenge  called the marshmallow challenge."
        chinese = "几年前，在这里的 TED 上，Peter Skillman 提出了一个名为“棉花糖挑战”的设计挑战。"
    else:
        english = src
        chinese = None

    # pad english input to max length 128
    src_ids = tokenizer.encode(english).ids + [tokenizer.token_to_id("<pad>")] * (
        128 - len(tokenizer.encode(english).ids)
    )
    src_ids = torch.tensor(src_ids, dtype=torch.int64)
    src_ids = src_ids.unsqueeze(0)
    src_mask = create_padding_mask(src_ids, pad_id=0)

    # decoder_input = torch.tensor(tgt_tokenizer.encode('<s>').ids, dtype = torch.int64).unsqueeze(0)
    sos_id = tokenizer.token_to_id("<s>")
    decoder_input = torch.tensor([[sos_id]], dtype=torch.int64)


    src_ids = src_ids.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    decoder_input = decoder_input.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    src_mask = src_mask.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    while decoder_input.size(1) < 128:
        with torch.no_grad():
            out = model(src_ids, decoder_input, src_pad_mask=src_mask)

        _, next_word = torch.max(out[:, -1], dim=-1)

        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(src_ids).fill_(next_word.item()).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))], dim=1
        )

        if next_word == tokenizer.token_to_id("</s>"):
            break

    output = tokenizer.decode(decoder_input[0].tolist(), skip_special_tokens=False)
    rich.print_color(f"English Input: {english}", color="cyan")
    rich.print_color(f"Model Output: {output}", color="cyan")
    if chinese is not None:
        rich.print_color(f"Reference: {chinese}", color="cyan")

    model.train()