"""
Реализация простой генеративной грамматики для шаблонизации реплик в скриптовой части чатбота.
"""

import random
import re


class TemplateNode(object):
    def __init__(self, next_node):
        self.next_node = next_node

    def run(self):
        raise NotImplementedError()


class TemplateNodeChar(TemplateNode):
    def __init__(self, char, next_node):
        super().__init__(next_node)
        self.char = char

    def run(self):
        if self.next_node:
            return self.char + self.next_node.run()
        else:
            return self.char

    def __repr__(self):
        if self.next_node:
            return self.char + self.next_node.__repr__()
        else:
            return self.char


class TemplateNodeChoice(TemplateNode):
    """
    Узел с несколькими вариантами содержимого, один из которых выбирается при каждой генерации
    (вариант1/вариант2/...)

    Каждый вариант может быть в свою очередь произвольным шаблонов, в том числе содержать свои альтернативы в (...)
    """
    def __init__(self, choices, next_node):
        super().__init__(next_node)
        self.choices = choices

    def run(self):
        choice_text = random.choice(self.choices).run()
        if self.next_node:
            return choice_text + self.next_node.run()
        else:
            return choice_text

    def __repr__(self):
        self_repr = '({})'.format('/'.join(map(str, self.choices)))
        if self.next_node:
            return self_repr + self.next_node.__repr__()
        else:
            return self_repr


class TemplateNodeCoalesce(TemplateNode):
    def __init__(self, optional_node, next_node):
        super().__init__(next_node)
        self.optional_node = optional_node

    def run(self):
        if random.random() > 0.5:
            optional_text = self.optional_node.run()
        else:
            optional_text = ''

        if self.next_node:
            return optional_text + self.next_node.run()
        else:
            return optional_text

    def __repr__(self):
        self_repr = '[{}]'.format(self.optional_node)
        if self.next_node:
            return self_repr + self.next_node.__repr__()
        else:
            return self_repr


class TemplateNodeNamedPattern(TemplateNode):
    def __init__(self, pattern_name, bound_pattern, next_node):
        super().__init__(next_node)
        self.pattern_name = pattern_name
        self.bound_pattern = bound_pattern

    def run(self):
        pattern_text = self.bound_pattern.run()
        if self.next_node:
            return pattern_text + self.next_node.run()
        else:
            return pattern_text

    def __repr__(self):
        if self.next_node:
            return self.pattern_name + self.next_node.__repr__()
        else:
            return self.pattern_name


class TemplatePattern(object):
    def __init__(self, pattern_str, named_patterns):
        self.pattern_str = pattern_str
        self.start_node = TemplatePattern.build_node(pattern_str, named_patterns)

    def __repr__(self):
        return self.pattern_str

    def run(self):
        return self.start_node.run()

    @staticmethod
    def build_node(pattern_str, named_patterns):
        if len(pattern_str) == 0:
            return None

        c1 = pattern_str[0]
        if c1 == '(':
            # Набор альтернатив внутри (...), разделенных символом /.
            # При генерации будет выбрана одна из этих альтернатив равновероятно.
            # Сейчас надо найти закрывающую ) с учетом возможной вложенности паттернов.
            i = 1
            n_open_rparens = 1
            l = len(pattern_str)
            closing_paren_found = False
            divider_positions = [0]
            while i < l:
                c = pattern_str[i]
                if c == '(':
                    n_open_rparens += 1
                elif c == ')':
                    n_open_rparens -= 1
                    if n_open_rparens == 0:
                        # Нашли закрывающую круглую скобку
                        closing_paren_found = True
                        divider_positions.append(i)
                        break
                elif c == '/':
                    if n_open_rparens == 1:
                        # разделитель альтернатив высшего уровня
                        divider_positions.append(i)
                i += 1

            if not closing_paren_found:
                raise ValueError('Closing parenthesis not found in pattern "{}"'.format(pattern_str))

            choices = []
            for i1, i2 in zip(divider_positions, divider_positions[1:]):
                choice_str = pattern_str[i1+1:i2]
                choice_node = TemplatePattern.build_node(choice_str, named_patterns)
                choices.append(choice_node)

            tail_str = pattern_str[divider_positions[-1]+1:]
            tail_node = TemplatePattern.build_node(tail_str, named_patterns)
            return TemplateNodeChoice(choices, tail_node)
        elif c1 == '$':
            # считываем имя паттерна после $ - символы латиницы/кириллицы или нижнего подчеркивания, до второго символа $
            pattern_name = re.search(r'^\$[a-zабвгдеёжзийклмнопрстуфхцчшщъыьэюя_][0-9a-zабвгдеёжзийклмнопрстуфхцчшщъыьэюя_]+\$', pattern_str, flags=re.I).group(0)
            if pattern_name not in named_patterns:
                raise ValueError('Missing named pattern "{}" in pattern "{}"'.format(pattern_name, pattern_str))
            next_node = TemplatePattern.build_node(pattern_str[len(pattern_name):], named_patterns)
            return TemplateNodeNamedPattern(pattern_name, named_patterns[pattern_name], next_node)
        elif c1 == '[':
            # Опциональный паттерн внутри [...]
            # Надо найти закрывающую скобку [, учитывая возможность вложенности
            i = 1
            n_open_sparens = 1
            l = len(pattern_str)
            closing_paren_found = False
            while i < l:
                c = pattern_str[i]
                if c == '[':
                    n_open_sparens += 1
                elif c == ']':
                    n_open_sparens -= 1
                    if n_open_sparens == 0:
                        # Нашли закрывающую квадратную скобку
                        closing_paren_found = True
                        break
                i += 1

            if not closing_paren_found:
                raise ValueError('Closing parenthesis not found in pattern "{}"'.format(pattern_str))

            optional_str = pattern_str[1:i]
            optional_node = TemplatePattern.build_node(optional_str, named_patterns)

            tail_str = pattern_str[i+1:]
            tail_node = TemplatePattern.build_node(tail_str, named_patterns)
            return TemplateNodeCoalesce(optional_node, tail_node)
        elif pattern_str.startswith('\\('):
            # Круглая скобочка заэкранирована, чтобы можно было вставлять в генерацию смайлики
            next_node = TemplatePattern.build_node(pattern_str[2:], named_patterns)
            return TemplateNodeChar("(", next_node=next_node)
        else:
            next_node = TemplatePattern.build_node(pattern_str[1:], named_patterns)
            return TemplateNodeChar(c1, next_node=next_node)


if __name__ == '__main__':
    named_patterns = dict()
    pattern_str = 'Кошка[-то] (ловит/ест) (мышку/птичку)'
    pattern = TemplatePattern(pattern_str, named_patterns)

    for _ in range(4):
        sample_text = pattern.run()
        print(sample_text)
