import os

from datasets import load_dataset

###### creating Dataset #######


def create_dataset(dataset, file_prefix, src="en", tgt="zh", debug=True):
    assert file_prefix is not None, "file_prefix must be set"
    
    if debug:
        dataset = dataset[: len(dataset) // 100]
        
    if debug:
        src_file_name = file_prefix + "src_debug" + ".txt"
        tgt_file_name = file_prefix + "tgt_debug" + ".txt"
        
    else:
        src_file_name = file_prefix + "src" + ".txt"
        tgt_file_name = file_prefix + "tgt" + ".txt"
        
        
    if os.path.exists(src_file_name) and os.path.exists(tgt_file_name):
        print(f"Files {src_file_name} and {tgt_file_name} already exist. skipping")
        
        return 
    
    with (
        open(src_file_name, "w", encoding = "utf-8") as src_f,
        open(tgt_file_name, "w", encoding = "utf-8")as tgt_f,
    ):
        
        for example in dataset:
            translation = example.get("translation", example)
            src_f.write(translation[src].strip() + "\n")
            tgt_f.write(translation[tgt].strip() + "\n")

    print(f"{src_file_name}, {tgt_file_name} saved")
    

if __name__ == "__main__":
    dataset = load_dataset(
        "Helsinki-NLP/opus-100",
        "en-zh",
    )

    create_dataset(dataset["train"], file_prefix="train", debug=False)
    create_dataset(dataset["test"], file_prefix="test", debug=False)
