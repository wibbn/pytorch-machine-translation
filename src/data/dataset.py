import torch
from typing import List, Tuple

from data.lang import Lang
from data.utils import TextUtils


class MtDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_lang: Lang,
        output_lang: Lang,
        training_pairs: List[Tuple[str, str]],
        max_len: str,
    ) -> None:
        """Dataset for machine translation

        Args:
            input_lang (Lang): input Lang to use
            output_lang (Lang): output lang to use
            training_pairs (List[Tuple[str, str]]): sentences pairs
            max_len (str): max len of sentence
        """
        self.training_pairs = training_pairs
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.training_pairs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.training_pairs[idx]
        tensor_pair = TextUtils.tensors_from_pair(
            pair=pair,
            max_len=self.max_len,
            input_lang=self.input_lang,
            output_lang=self.output_lang,
            eos_token=self.input_lang.word2index["[EOS]"],
        )
        return tensor_pair[0], tensor_pair[1]
