import re

from ruchatbot.utils.chunk_tools import normalize_chunk


class SayingPhraseEntry:
    """Разобранный на элементы подстановочный терм в выводимой say фразе"""
    def __init__(self, name, raw_text, tags):
        self.raw_text = raw_text
        self.name = name
        self.tags = tags


class SayingPhrase:
    def __init__(self, phrase_str):
        self.raw_text = phrase_str
        self.name2entry = dict()
        for slot_prefix in ['NP', 'VI', 'AP', 'VP']:
            if '$'+slot_prefix in phrase_str:
                for m in re.finditer(r'\$(' + slot_prefix + r'\d+)', phrase_str):
                    entry_text = m.group(0)
                    entry_name = m.group(1)
                    entry_tags = None

                    args_pos = m.span()[1]
                    if args_pos <= len(phrase_str)-1 and phrase_str[args_pos] == '(':
                        args_start = args_pos + 1
                        args_end = phrase_str.index(')', args_start)
                        entry_text = phrase_str[m.span()[0]:args_end+1]

                        args_str = phrase_str[args_start:args_end]
                        entry_tags = [a.strip() for a in args_str.strip().split(',')]

                    entry = SayingPhraseEntry(entry_name, entry_text, entry_tags)
                    self.name2entry[entry_name] = entry

    def has_entries(self):
        return len(self.name2entry) > 0


def substitute_bound_variables(phrase, condition_matching_results, text_utils):
    assert(isinstance(phrase, SayingPhrase))

    utterance = phrase.raw_text

    # Если нужно сделать подстановку сматченных при проверке условия чанков.
    if condition_matching_results and condition_matching_results.has_groups() and phrase.has_entries():
        for name, group in condition_matching_results.groups.items():
            group_ancor = name.upper()
            if group_ancor in phrase.name2entry:
                entry = phrase.name2entry[group_ancor]
                words = group.words

                # Нужно просклонять чанк?
                if entry.tags:
                    tokens = group.phrase_tokens
                    target_tags = dict()
                    for tag in entry.tags:
                        if tag in ('ИМ', 'ВИН', 'РОД', 'ТВОР', 'ДАТ', 'ПРЕДЛ'):
                            target_tags['ПАДЕЖ'] = tag
                        elif tag in ('ЕД', 'МН'):
                            target_tags['ЧИСЛО'] = tag
                        else:
                            raise NotImplementedError()

                    words = normalize_chunk(tokens, edges=None, flexer=text_utils.flexer,
                                            word2tags=text_utils.word2tags, target_tags=target_tags)

                # Подставляем слова чанка вместо подстроки $NP1(...)
                entry_value = ' '.join(words)
                utterance = utterance.replace(entry.raw_text, entry_value)

    return utterance
