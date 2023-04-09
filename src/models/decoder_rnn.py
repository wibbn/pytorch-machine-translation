import torch

class DecoderRNN(torch.nn.Module):
    def __init__(self, embedding_size: int, decoder_vocab_size: int) -> None:
        super(DecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = torch.nn.Embedding(decoder_vocab_size, embedding_size)
        self.gru = torch.nn.GRU(embedding_size, embedding_size)

    def forward(self, input, hidden, batch_size):
        output = self.embedding(input).reshape(1, batch_size, self.embedding_size)
        output, hidden = self.gru(output, hidden)
        return output, hidden