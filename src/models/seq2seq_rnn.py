import torch
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Tuple, List
from torchtext.data.metrics import bleu_score

from models.decoder_rnn import DecoderRNN
from models.encoder_rnn import EncoderRNN
from models.attention import Seq2seqAttention


class Seq2SeqRNN(pl.LightningModule):
    def __init__(
            self,
            encoder_vocab_size: int,
            encoder_embedding_size: int,
            decoder_embedding_size: int,
            decoder_vocab_size: int,
            lr: float,
            output_lang_index2word: Dict[int, str],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.output_lang_index2word = output_lang_index2word
        self.encoder = EncoderRNN(
            encoder_vocab_size=encoder_vocab_size, embedding_size=encoder_embedding_size
        )
        self.attention_module = Seq2seqAttention()
        self.decoder = DecoderRNN(
            embedding_size=decoder_embedding_size, decoder_vocab_size=decoder_vocab_size
        )

        self.vocab_projection_layer = torch.nn.Linear(2 * decoder_embedding_size, decoder_vocab_size)

        self.criterion = torch.nn.NLLLoss()
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]
        encoder_states, encoder_last_hidden = self.encoder(input_tensor)
        decoder_hidden = encoder_last_hidden
        decoder_input = torch.tensor(
            [[0] * batch_size], dtype=torch.long, device=self.device
        ).view(1, batch_size, 1)
        predicted = []
        decoder_outputs = []
        for _ in range(input_tensor.shape[1]):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, batch_size
            )
            weighted_decoder_output = self.attention_module(decoder_output.squeeze(dim=0), encoder_states)
            decoder_output = torch.cat([decoder_output.squeeze(dim=0), weighted_decoder_output], dim=1)
            linear_vocab_proj = self.vocab_projection_layer(decoder_output)
            target_vocab_distribution = self.softmax(linear_vocab_proj)
            _, topi = target_vocab_distribution.topk(1)
            predicted.append(topi.clone().detach().cpu())
            decoder_input = topi.reshape(1, batch_size, 1)
            decoder_outputs.append(target_vocab_distribution.squeeze())
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
