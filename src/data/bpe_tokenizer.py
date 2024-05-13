from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class BPETokenizer:
    def __init__(self, sentence_list, vocab_size=4000, pad_flag=True, max_sent_len=15):
        """
        sentence_list - список предложений для обучения
        """
        self.pad_flag = pad_flag
        self.max_sent_len = max_sent_len

        self.special_tokens = ["UNK", "PAD", "SOS", "EOS"]

        self.tokenizer = Tokenizer(BPE(unk_token="UNK"))
        trainer = BpeTrainer(vocab_size=vocab_size,
                             special_tokens=self.special_tokens)

        self.tokenizer.pre_tokenizer = Whitespace()

        self.tokenizer.train_from_iterator(sentence_list, trainer=trainer)

        self.tokenizer.post_processor = TemplateProcessing(
            single="SOS $A EOS",
            pair="SOS $A EOS $B:1 EOS:1",
            special_tokens=[("SOS", self.tokenizer.token_to_id("SOS")),
                            ("EOS", self.tokenizer.token_to_id("EOS"))],
        )

    def pad_sent(self, token_ids_list):
        token_len = 2 * self.max_sent_len # сделаем длину токенизированного предложения в 2 раза брольше
        if len(token_ids_list) < token_len:
            padded_token_ids_list = token_ids_list + [self.tokenizer.token_to_id("PAD")] * (
                    token_len - len(token_ids_list))
        else:
            padded_token_ids_list = token_ids_list[:token_len - 1] + [self.tokenizer.token_to_id("EOS")]
        return padded_token_ids_list

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        tokenized_data = self.tokenizer.encode(sentence).ids
        if self.pad_flag:
            tokenized_data = self.pad_sent(tokenized_data)
        return tokenized_data

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        decoded = self.tokenizer.decode(token_list)
        tokens = decoded.split()
        res = []
        for token in tokens:
            if token in self.special_tokens:
                continue
            res.append(token)
        return res


if __name__ == '__main__':
    pass