import os

import pyconll
from ufal.udpipe import Model, Pipeline, ProcessingError


class UDPipeToken:
    def __init__(self, ud_token, upos=None, tags=None):
        self.id = ud_token.id
        self.form = ud_token.form
        self.upos = ud_token.upos if upos is None else upos
        self.lemma = ud_token.lemma
        self.tags = [(k + '=' + list(vx)[0]) for k, vx in ud_token.feats.items()] if tags is None else list(tags)
        self.deprel = ud_token.deprel
        self.head = ud_token.head

    def __repr__(self):
        return self.form

    def get_attr(self, attr_name):
        k = attr_name + '='
        for t in self.tags:
            if t.startswith(k):
                return t.split('=')[1]
        return ''


def get_attr(token, tag_name):
    if tag_name in token.feats:
        v = list(token.feats[tag_name])[0]
        return v

    return ''


class Parsing(object):
    def __init__(self, tokens, text):
        self.tokens = list(tokens)
        self.text = text

    def get_text(self):
        return self.text

    def __repr__(self):
        return self.text

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return list(self.tokens).__iter__()

    def __getitem__(self, i):
        return self.tokens[int(i)-1]


class UdpipeParser:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.error = None

    def load(self, model_path):
        if os.path.isfile(model_path):
            udp_model_file = model_path
        else:
            udp_model_file = os.path.join(model_path, 'udpipe_syntagrus.model')

        self.model = Model.load(udp_model_file)
        self.pipeline = Pipeline(self.model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        self.error = ProcessingError()

    def parse_text(self, text):
        parsings = []

        processed = self.pipeline.process(text, self.error)
        if self.error.occurred():
            return None
        try:
            for parsing0 in pyconll.load_from_string(processed):
                parsing = []
                for itoken, token in enumerate(parsing0):
                    utoken = token.form.lower()

                    # 24-12-2021 Руками исправляем некоторые очень частотные ошибки.
                    if utoken == 'душе':
                        is_soul_dative = False
                        if token.id == '1':
                            is_soul_dative = True
                        else:
                            for neighb_token in parsing0[itoken-1: itoken+2]:
                                if neighb_token.upos in ('ADJ', 'DET') and get_attr(neighb_token, 'Gender') == 'Fem':
                                    is_soul_dative = True
                                    break

                        if is_soul_dative:
                            parsing.append(UDPipeToken(token, upos='NOUN', tags=['Case=Dat']))
                            continue
                    if utoken in ['чтоб']:
                        # Исправляем ошибки разметки некоторых слов в UDPipe.Syntagrus
                        parsing.append(UDPipeToken(token, upos='SCONJ', tags=[]))
                    elif utoken in ['средь']:
                        parsing.append(UDPipeToken(token, upos='ADP', tags=[]))
                    else:
                        parsing.append(UDPipeToken(token))
                parsings.append(Parsing(parsing, parsing0.text))
        except:
            return None

        return parsings


if __name__ == '__main__':
    parser = UdpipeParser()
    parser.load('/home/inkoziev/polygon/text_generator/models')

    parsing = parser.parse_text('Твоей душе испорченной')[0]
    for token in parsing:
        print('{} {} {}'.format(token.form, token.upos, token.tags))


