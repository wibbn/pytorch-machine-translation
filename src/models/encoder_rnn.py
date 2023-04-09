import torch

class EncoderRNN(torch.nn.Module):
    def __init__(self, encoder_vocab_size: int, embedding_size: int) -> None:
        super(EncoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = torch.nn.Embedding(encoder_vocab_size, embedding_size)
        self.gru = torch.nn.GRU(embedding_size, embedding_size, batch_first=True)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, input):
        embedded = self.embedding(input).squeeze()
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded)
        return output, hidden