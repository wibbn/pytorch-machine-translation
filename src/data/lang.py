class Lang:
    """
    Class for storing tokens
    """

    def __init__(self, name: str):
        """
        Args:
            name (str): string name of the lang. Influences nothing.
        """
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "[SOS]", 1: "[EOS]"}
        self.word2index = {"[SOS]": 0, "[EOS]": 1}
        self.n_words = 2

    def add_sentence(self, sentence: str):
        """Add words in sentence to vocab

        Args:
            sentence (str): sentence
        """
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word: str):
        """add word to vocab

        Args:
            word (str): word to add
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def __str__(self):
        return "Class: Lang\nName: {}\nn_words: {}".format(self.name, self.n_words)
