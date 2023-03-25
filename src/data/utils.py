import re
import torch
from tqdm import tqdm
from typing import List, Tuple, Optional

from data.lang import Lang


class TextUtils:
    @staticmethod
    def normalize_text(s: str) -> str:
        """Normalizes string, removes punctuation and
        non alphabet symbols

        Args:
            s (str): string to mormalize

        Returns:
            str: normalized string
        """
        s = s.lower()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Zа-яйёьъА-Яй]+", r" ", s)
        s = s.strip()
        return s

    @staticmethod
    def pad(inputs: List[str], length: int, eos_token: str) -> List[str]:
        """Pad (or trim) input indices to length

        Args:
            inputs (List[str]): list of words indices
            length (int): max length
            eos_token (str): eos token

        Returns:
            List[str]: list of padded indices
        """
        if len(inputs) < length:
            inputs += [eos_token] * (length - len(inputs))
        else:
            inputs = inputs[:length]
            inputs[-1] = eos_token
        return inputs

    @staticmethod
    def indexes_from_sentence(lang: Lang, sentence: str) -> List[str]:
        """convert words to its indices using Lang

        Args:
            lang (Lang): lang to use
            sentence (str): string sentence

        Returns:
            List[str]: list of indices
        """
        sos_token = [lang.word2index["[SOS]"]]
        sentence_tokens = [lang.word2index[word] for word in sentence.split(" ")]
        return sos_token + sentence_tokens

    @staticmethod
    def tensor_from_sentence(
        lang: Lang, sentence: str, max_len: int, eos_token: str
    ) -> torch.Tensor:
        """convert indices to tensor

        Args:
            lang (Lang): Lang
            sentence (str): string sentence
            max_len (int): max len of sententence in tokens
            eos_token (str): end of string token

        Returns:
            torch.Tensor: tensor from sentence
        """
        indexes = TextUtils.pad(
            TextUtils.indexes_from_sentence(lang, sentence), max_len, eos_token
        )
        return torch.tensor(indexes, dtype=torch.long, device="cpu").view(-1, 1)

    @staticmethod
    def tensors_from_pair(
        pair: Tuple[str, str],
        max_len: int,
        input_lang: Lang,
        output_lang: Lang,
        eos_token: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts pair of sentneces to pair of tensors

        Args:
            pair (Tuple[str, str]): pair
            max_len (int): maximum
            input_lang (Lang): input lang
            output_lang (Lang): output lang
            eos_token (str): end of string token

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: pair of tensors from sentences
        """
        input_tensor = TextUtils.tensor_from_sentence(
            input_lang, pair[0], max_len, eos_token
        )
        target_tensor = TextUtils.tensor_from_sentence(
            output_lang, pair[1], max_len, eos_token
        )
        return input_tensor, target_tensor

    @staticmethod
    def read_langs_pairs_from_file(
        filename: str, lang1: str, lang2: str, reverse: Optional[bool]
    ) -> Tuple[Lang, Lang, List[Tuple[str, str]]]:
        """Read lang from file

        Args:
            filename (str): path to dataset
            lang1 (str): name of first lang
            lang2 (str): name of second lang
            reverse (Optional[bool]): revers inputs (eng->ru of ru->eng)

        Returns:
            Tuple[Lang, Lang, List[Tuple[str, str]]]: tuple of
                (input lang class, out lang class, string pairs)
        """
        lang_pairs = []
        with open(filename, mode="r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        for line in tqdm(lines, desc="Reading from file"):
            lang_pair = tuple(map(TextUtils.normalize_text, line.split("\t")[:2]))
            lang_pairs.append(lang_pair)

        if reverse:
            lang_pairs = [tuple(reversed(pair)) for pair in lang_pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)

        return input_lang, output_lang, lang_pairs
