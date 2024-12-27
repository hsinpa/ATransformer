import torch
from torch.utils.data import Dataset


import torch
from tiktoken import Encoding
from torch.utils.data import Dataset, DataLoader

# {'<|fim_prefix|>', '<|endoftext|>', '<|endofprompt|>', '<|fim_middle|>', '<|fim_suffix|>'}
SpecialToken_EndOfText = '<|endoftext|>'

def load_data_full_text(relative_path):
    text = ""
    with open(relative_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text

class TextDataSet(Dataset):
    def __init__(self, file_paths: list[str], tokenizer: Encoding, max_length: int, stride: int):
        self._tokenizer = tokenizer
        self._input_ids = []
        self._target_ids = []

        full_encode_array: list[int] = []

        for file_path in file_paths:
            full_text = load_data_full_text(file_path)
            full_text += SpecialToken_EndOfText

            encode_array = tokenizer.encode(full_text, allowed_special={SpecialToken_EndOfText})
            full_encode_array.extend(encode_array)

        for i in range(0, len(full_encode_array) - max_length, stride):
            input_chunk = full_encode_array[i:i + max_length]
            target_chunk = full_encode_array[i + 1: i + max_length + 1]
            self._input_ids.append(torch.tensor(input_chunk))
            self._target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self._input_ids)

    def __getitem__(self, idx):
        return self._input_ids[idx], self._target_ids[idx]


def get_data_loader(file_paths: list[str], tokenizer: Encoding, max_length: int, stride: int, batch_size: int,
                    shuffle: bool= True, drop_last: bool = False) -> DataLoader:
    text_dataset = TextDataSet(file_paths, tokenizer, max_length, stride)
    return DataLoader(text_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
