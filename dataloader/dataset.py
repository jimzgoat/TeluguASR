import sys
sys.path.append("../")
import torch

from utils.feature import load_wav
from typing import Dict

# class DefaultCollate:
#     def __init__(self, processor, sr) -> None:
#         self.processor = processor
#         self.sr = sr
#     def __call__(self, inputs) -> Dict[str, torch.tensor]:
#         features, transcripts = zip(*inputs)
#         features, transcripts = list(features), list(transcripts)
#         batch = self.processor(features, sampling_rate=16000, padding="longest", return_tensors="pt")

#         with self.processor.as_target_processor():
#             labels_batch = self.processor(transcripts, padding="longest", return_tensors="pt")

#         batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

#         return batch


class DefaultCollate:
    def __init__(self, processor, sr) -> None:
        self.processor = processor
        self.sr = sr

    def __call__(self, inputs) -> Dict[str, torch.tensor]:
        features, transcripts = zip(*inputs)
        features, transcripts = list(features), list(transcripts)

        # Process features (audio) with padding to the longest sequence in the batch
        batch = self.processor(features, sampling_rate=self.sr, padding="longest", return_tensors="pt")

        with self.processor.as_target_processor():
            # Tokenize the transcripts with padding and truncation to ensure consistent lengths
            labels_batch = self.processor.tokenizer(
                transcripts,
                padding="longest",  # Pad to the longest transcript in the batch
                truncation=True,  # Truncate sequences that are too long
                return_tensors="pt"
            )

        # Set the labels to -100 where attention_mask is 0 (i.e., padding tokens in the labels)
        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        return batch
        

class Dataset:
    def __init__(self, data, sr, preload_data, transform = None):
        self.data = data
        self.sr = sr
        self.transform = transform
        self.preload_data = preload_data
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx) -> tuple:
        item = self.data.iloc[idx]
        if not self.preload_data:
            feature = load_wav(item['path'], sr = self.sr)
        else:
            feature = item['wav']
        
        return feature, item['transcript']

