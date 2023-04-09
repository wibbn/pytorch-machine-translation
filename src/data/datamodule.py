import torch
import pytorch_lightning as pl
from typing import Optional
from sklearn.model_selection import train_test_split

from data.dataset import MtDataset
from data.utils import TextUtils


class DataManager(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_lang_n_words = None
        self.output_lang_n_words = None

    def prepare_data(self) -> None:
        self.input_lang, self.output_lang, pairs = TextUtils.read_langs_pairs_from_file(
            filename=self.config["filename"],
            lang1=self.config["lang1"],
            lang2=self.config["lang2"],
            reverse=self.config["reverse"],
        )

        pairs = list(filter(self.config["filter"], pairs))
        for pair in pairs:
            self.input_lang.add_sentence(pair[0])
            self.output_lang.add_sentence(pair[1])

        if self.config["quantile"] is not None:
            comparator = lambda x: max(len(x[0].split(" ")), len(x[1].split(" ")))
            pairs = sorted(pairs, key=comparator)[
                : int(len(pairs) * self.config["quantile"])
            ]

        self.train_data, self.val_data = train_test_split(
            pairs, train_size=self.config["train_size"]
        )

        self.max_len = max(
            [max(len(el[0].split(" ")), len(el[1].split(" "))) for el in pairs]
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = MtDataset(
            self.input_lang, self.output_lang, self.train_data, self.max_len
        )
        self.val_dataset = MtDataset(
            self.input_lang, self.output_lang, self.val_data, self.max_len
        )
        self.input_lang_n_words = self.train_dataset.input_lang.n_words
        self.output_lang_n_words = self.train_dataset.output_lang.n_words

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.config["num_workers"],
            batch_size=self.config["batch_size"],
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.config["num_workers"],
            batch_size=self.config["batch_size"],
        )
