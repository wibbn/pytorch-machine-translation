import torch
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Tuple, List
from torchtext.data.metrics import bleu_score


class EncoderRNN(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int) -> None:
        super(EncoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.gru = torch.nn.GRU(embedding_size, embedding_size, batch_first=True)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, input):
        embedded = self.embedding(input).squeeze()
        embedded = self.dropout(embedded)
        _, hidden = self.gru(embedded)
        return hidden


class DecoderRNN(torch.nn.Module):
    def __init__(self, embedding_size: int, output_size: int) -> None:
        super(DecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = torch.nn.Embedding(output_size, embedding_size)
        self.gru = torch.nn.GRU(embedding_size, embedding_size)
        self.out = torch.nn.Linear(embedding_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, batch_size):
        output = self.embedding(input).reshape(1, batch_size, self.embedding_size)
        output, hidden = self.gru(output, hidden)
        linear_proj = self.out(output)
        output = self.softmax(linear_proj)
        return output, hidden


class Seq2SeqRNN(pl.LightningModule):
    def __init__(
        self,
        encoder_vocab_size: int,
        encoder_embedding_size: int,
        decoder_embedding_size: int,
        decoder_output_size: int,
        lr: float,
        output_lang_index2word: Dict[int, str],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.output_lang_index2word = output_lang_index2word
        self.encoder = EncoderRNN(
            vocab_size=encoder_vocab_size, embedding_size=encoder_embedding_size
        )
        self.decoder = DecoderRNN(
            embedding_size=decoder_embedding_size, output_size=decoder_output_size
        )

        self.criterion = torch.nn.NLLLoss()

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        decoder_hidden = (encoder_hidden := self.encoder(input_tensor))
        decoder_input = torch.tensor(
            [[0] * batch_size], dtype=torch.long, device=self.device
        ).view(1, batch_size, 1)
        predicted = []
        decoder_outputs = []
        for _ in range(input_tensor.shape[1]):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, batch_size
            )
            _, topi = decoder_output.topk(1)
            predicted.append(topi.clone().detach().cpu())
            decoder_input = topi.reshape(1, batch_size, 1)
            decoder_outputs.append(decoder_output.squeeze())
        return predicted, decoder_outputs

    def training_step(self, batch, batch_idx):
        input_tensor, target_tensor = batch
        predicted, decoder_outputs = self(input_tensor)

        # Calculate Loss
        target_length = target_tensor.shape[1]
        loss = 0
        for di in range(target_length):
            loss += self.criterion(
                decoder_outputs[di].squeeze(), target_tensor[:, di, :].squeeze()
            )
        loss = loss / target_length

        # Calculate BLEU
        predicted = torch.stack(predicted)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        batch_bleu, _, _ = self.calculate_batch_bleu(predicted, actuals)

        # Log
        self.log("train_bleu", batch_bleu, on_epoch=False, on_step=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_tensor, target_tensor = batch
        predicted, decoder_outputs = self(input_tensor)

        # Calculate Loss
        target_length = target_tensor.shape[1]
        loss = 0
        for di in range(target_length):
            loss += self.criterion(
                decoder_outputs[di].squeeze(), target_tensor[:, di, :].squeeze()
            )
        loss = loss / target_length

        # Calculate BLEU
        predicted = torch.stack(predicted)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        batch_bleu, actual_sentences, predicted_sentences = self.calculate_batch_bleu(
            predicted, actuals
        )
        if batch_idx == 0:
            for a, b in zip(actual_sentences[:15], predicted_sentences[:15]):
                print(f"{a} ---> {b}")

        # Log
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val_bleu", batch_bleu, on_step=True, on_epoch=True, prog_bar=False)
        return {"val_loss": loss, "val_bleu": torch.tensor(batch_bleu)}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def calculate_batch_bleu(
        self, predicted: np.ndarray, actual: np.ndarray
    ) -> Tuple[float, List[str], List[str]]:
        """Convert predictions to sentences and calculate
        BLEU score.

        Args:
            predicted (np.ndarray): batch of indices of predicted words
            actual (np.ndarray): batch of indices of ground truth words

        Returns:
            Tuple[float, List[str], List[str]]: tuple of
                (
                    bleu score,
                    ground truth sentences,
                    predicted sentences
                )
        """
        batch_bleu = []
        predicted_sentences = []
        actual_sentences = []
        for a, b in zip(predicted, actual):
            words_predicted = []
            words_actual = []
            for a_i, b_i in zip(a, b):
                words_predicted.append(self.output_lang_index2word[a_i])
                words_actual.append(self.output_lang_index2word[b_i])
            words_predicted = list(
                filter(lambda x: x != "[SOS]" and x != "[EOS]", words_predicted)
            )
            words_actual = list(
                filter(lambda x: x != "[SOS]" and x != "[EOS]", words_actual)
            )
            batch_bleu.append(bleu_score([words_predicted], [[words_actual]]))
            predicted_sentences.append(" ".join(words_predicted))
            actual_sentences.append(" ".join(words_actual))
        batch_bleu = np.mean(batch_bleu)
        return batch_bleu, actual_sentences, predicted_sentences
