import torch
import os


def encode_file_pt(
    tokenizer,
    in_path :str,
    out_path :str,
    add_special_tokens: bool = True,
    bos_token: str = "[CLS]",
    eos_token: str = "[SEP]",
    max_lines: int|None = None,
):
    if os.path.exists(out_path):
        print(f"{out_path} already exists")
        return

    bos_id = tokenizer.token_to_id(bos_token)
    eos_id = tokenizer.token_to_id(eos_token)
    if bos_id is None or eos_id is None:
        raise ValueError(
            f"Tokenizer must define {bos_token!r} and {eos_token!r} tokens"
        )

    all_ids = []
    with open(in_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            text = line.rstrip("\r\n")
            enc = tokenizer.encode(text)
            ids = enc.ids
            if add_special_tokens:
                ids = [bos_id] + ids + [eos_id]
            all_ids.append(torch.tensor(ids, dtype=torch.long))

    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(all_ids, out_path)
    print(f"saved {len(all_ids)} sequences to {out_path}")