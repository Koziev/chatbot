""" Структура для результатов токенизации, морфологического и синтаксического разбора запросной строки """


def extract_constit0(tokens, token):
    tx = [token]
    for t in tokens:
        if t.head == token.id:
            tx.extend(extract_constit0(tokens, t))
    return tx


class ParsingResult:
    def __init__(self, tokens, text):
        self.text = text
        #self.root_token = None
        #self.tokens = None
        #self.lemmas = None
        self.udp_tokens = list(tokens)
        #self.top_constituents = None

    def __len__(self):
        return len(self.udp_tokens)

    def extract_constituent(self, root_token):
        tx = extract_constit0(self.udp_tokens, root_token)
        return sorted(tx, key=lambda z: int(z.id))

    def get_word(self, index):
        return self.udp_tokens[index].form

    def get_lemma(self, index):
        return self.udp_tokens[index].lemma

    def get_token(self, index):
        return self.udp_tokens[index]

    def get_words(self, start_index, nb_words):
        return [self.udp_tokens[i].form for i in range(start_index, start_index+nb_words)]

    def get_lemmas(self, start_index, nb_words):
        return [self.udp_tokens[i].lemma for i in range(start_index, start_index+nb_words)]
