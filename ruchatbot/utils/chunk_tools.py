

def normalize_chunk(tokens, edges, flexer, word2tags, target_tags=dict()):
    if len(tokens) == 0:
        return []

    normal_tokens = [t.word for t in tokens]

    # Ищем среди токенов чанка ведущий токен.
    # По дефолту - ищем первое существительное
    uppermost_token = tokens[-1]
    for token in tokens:
        if token.tagset.startswith('NOUN'):
            uppermost_token = token
            break

    if False:
        # Определяем глубину каждого токена в синтаксическом дереве
        itoken2depth = find_token_tree_depths(edges, tokens)

        uppermost_depth = itoken2depth[uppermost_token.word_index]
        for token in tokens:
            if itoken2depth[token.word_index] < uppermost_depth:
                if uppermost_token.tagset.split('|')[0] in ('NOUN', 'PRON'):
                    uppermost_depth = itoken2depth[token.word_index]
                    uppermost_token = token

    head_noun_tags = uppermost_token.tagset.split('|')
    # Если найденный корень чанка является существительным...
    if head_noun_tags[0] in ('NOUN', 'PRON'):
        # Будем склонять это главное существительное, а потом пересогласуем все зависимые прилагательные.
        pass

    target_case = target_tags.get('ПАДЕЖ', 'ИМ')
    target_number = target_tags.get('ЧИСЛО', None)
    if not target_number:
        if 'Number=Plur' in head_noun_tags:
            target_number = 'МН'
        else:
            target_number = 'ЕД'

    head_noun_forms = list(flexer.find_forms_by_tags(uppermost_token.lemma, [(u'ПАДЕЖ', target_case), (u'ЧИСЛО', target_number)]))
    if head_noun_forms:
        head_noun = head_noun_forms[0]

        new_forms = dict()
        new_forms[uppermost_token.word_index] = head_noun

        # пересогласовать зависимые прилагательные
        adjs = set()

        # для простоты реализации сейчас считаем, что все прилагательные слева от существительного
        # зависят от него и нуждаются в пересогласовании.
        for token in tokens:
            if token == uppermost_token:
                break

            if token.tagset.startswith('ADJ'):
                adjs.add(token)

        # добавляем прилагательные, которые в синтаксическом дереве зависят от главного
        # существительного.
        if edges:
            for token in tokens:
                if token.tagset.startswith('ADJ'):
                    for edge in edges:
                        if edge[1] == token.word_index and edge[3] == uppermost_token.word_index:
                            adjs.add(token)
                            break

        gender = None
        if target_number == 'ЕД':
            for tag in uppermost_token.tagset.split('|'):
                if tag.startswith('Gender'):
                    gender = tag.split('=')[1]
                    if gender == 'Fem':
                        gender = 'ЖЕН'
                    elif gender == 'Masc':
                        gender = 'МУЖ'
                    else:
                        gender = 'Neut'
                    break

        adj_anymacy = None
        if target_case == 'ВИН':
            noun_tagsets = word2tags[uppermost_token.lemma]
            for tagset in noun_tagsets:
                tagset = tagset.split(' ')
                if tagset[0] == 'СУЩЕСТВИТЕЛЬНОЕ':
                    tagset = dict(x.split('=') for x in tagset[1:])
                    if target_number == 'МН':
                        adj_anymacy = ('ОДУШ', tagset['ОДУШ'])
                    else:
                        if tagset['РОД'] == 'МУЖ':
                            adj_anymacy = ('ОДУШ', tagset['ОДУШ'])
                    break

        for token in adjs:
            adj_tags = [('ПАДЕЖ', target_case), ('ЧИСЛО', target_number), ('КРАТКИЙ', '0'), ('СТЕПЕНЬ', 'АТРИБ')]

            if adj_anymacy:
                adj_tags.append(adj_anymacy)

            if gender:
                adj_tags.append(('РОД', gender))

            adj_forms = list(flexer.find_forms_by_tags(token.lemma, adj_tags))
            if adj_forms:
                adj_form = adj_forms[0]
                new_forms[token.word_index] = adj_form
            else:
                new_forms[token.word_index] = token.lemma

        normal_tokens = []
        for token in tokens:
            normal_token = new_forms.get(token.word_index, token.word)
            normal_tokens.append(normal_token)

    return normal_tokens


def find_token_tree_depths(edges, tokens):
    itoken2depth = dict()

    if edges:
        # ищем корень
        for edge in edges:
            if edge[2] is None:
                # Корень найден, от него спускаем вниз по веткам
                itoken2depth[edge[1]] = 0
                find_token_tree_depths2(edges, edge[1], 0, itoken2depth)

    # если корень не найден, то сделаем дефолтную инициализацию уровней по токенам
    # слева направо.
    if len(itoken2depth) == 0:
        if edges:
            max_index = max(edge[1] for edge in edges)
        else:
            max_index = max(token.word_index for token in tokens)

        itoken2depth = dict((i, i) for i in range(max_index+1))

    return itoken2depth


def find_token_tree_depths2(edges, parent_index, parent_depth, itoken2depth):
    for edge in edges:
        if edge[3] == parent_index:
            itoken = edge[1]
            itoken2depth[itoken] = parent_depth + 1
            find_token_tree_depths2(edges, itoken, parent_depth + 1, itoken2depth)
