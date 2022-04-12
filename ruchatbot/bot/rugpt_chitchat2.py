"""
Обертка для генеративной модели читчата и некоторых других функций в чатботе.

04-03-2022 Сдеоланы отдельные методы generate_autoquestions, generate_chitchat, generate_confabulations,
           generate_interpretations, формирующие внутри себя правильные форматы входных данных для генеративной
           модели.
"""

import logging
import math

import torch

from ruchatbot.bot.rugpt_base import RugptBase


class RugptChitchat(RugptBase):
    def __init__(self):
        super(RugptChitchat, self).__init__()
        self.beam_k = 50
        self.beam_p = 0.9
        self.temperature = 1.0

    def load(self, model_path):
        logging.debug('Start loading generative model from "%s"', model_path)
        self.load_from_path(model_path)

    def generate_chitchat(self, context_replies, num_return_sequences):
        outputs = set()

        input_dialog = []
        for r in context_replies:
            if r.startswith('[') or r.startswith('{'):
                # Специальные метки в квадратных скобочках
                input_dialog.append(r)
            elif r.startswith('-'):
                # Обычные реплики, в начале которых уже стоит тире
                input_dialog.append(r)
            else:
                # Обычные реплики без начального "- "
                input_dialog.append('- ' + r)

        prompt_text = '<s>{chitchat}\n' + '\n'.join(input_dialog) + '\n'
        raw_outputs = self.generate_output_from_prompt(prompt_text, num_return_sequences, temperature=self.temperature)
        for o in raw_outputs:
            lines = o.split('\n')
            line1 = lines[0].strip()
            if line1.startswith('-'):
                line1 = line1[1:].strip()
            outputs.add(line1)

        return list(outputs)

    def score_dialogues(self, dialogues):
        # из-за разной длины текстов придется выполнять вычисления по 1 тексту за раз :(
        scores = []
        for dialog in dialogues:
            encoded_text = self.tokenizer.encode('<s>{chitchat}\n' + '\n'.join(dialog))
            t = torch.tensor(encoded_text, dtype=torch.long, device=self.device).unsqueeze(0)
            with torch.no_grad():
                loss = self.model(t, labels=t)
            #perplexity = math.exp(loss[0].item())
            score = loss[0].item()
            scores.append(math.exp(-score))

        return scores

    def generate_autoquestions(self, context_replies, num_return_sequences):
        outputs = set()

        input_dialog = []
        for r in context_replies:
            if r.startswith('[') or r.startswith('{'):
                input_dialog.append(r)
            elif r.startswith('-'):
                # Обычные реплики, в начале которых уже стоит тире
                input_dialog.append(r)
            else:
                # Обычные реплики без начального "- "
                input_dialog.append('- ' + r)

        prompt_text = '<s>{autoquestion}\n' + '\n'.join(input_dialog) + '\n'
        raw_outputs = self.generate_output_from_prompt(prompt_text, num_return_sequences, temperature=self.temperature)
        for o in raw_outputs:
            lines = o.split('\n')
            line1 = lines[0].strip()
            if line1.startswith('-'):
                line1 = line1[1:].strip()
            outputs.add(line1)

        return list(outputs)

    def generate_confabulations(self, context_replies, num_return_sequences):
        outputs = set()

        input_dialog = []
        for r in context_replies:
            if r.startswith('[') or r.startswith('{'):
                input_dialog.append(r)
            elif r.startswith('-'):
                # Обычные реплики, в начале которых уже стоит тире
                input_dialog.append(r)
            else:
                # Обычные реплики без начального "- "
                input_dialog.append('- ' + r)

        prompt_text = '<s>{confabulation}\n' + '\n'.join(input_dialog) + '\n'
        raw_outputs = self.generate_output_from_prompt(prompt_text, num_return_sequences, temperature=self.temperature)
        for o in raw_outputs:
            lines = o.split('\n')
            line1 = lines[0].strip()
            if line1.startswith('-'):
                line1 = line1[1:].strip()
            outputs.add(line1)

        return list(outputs)

    def generate_interpretations(self, context_replies, num_return_sequences):
        input_context = []
        for r in context_replies:
            if r.startswith('[') or r.startswith('{'):
                input_context.append(r)
            elif r.startswith('-'):
                # Обычные реплики, в начале которых уже стоит тире
                input_context.append(r)
            else:
                # Обычные реплики без начального "- "
                input_context.append('- ' + r)

        prompt_text = '<s>' + '\n'.join(input_context) + ' #'
        outputs = self.generate_output_from_prompt(prompt_text, num_return_sequences, temperature=self.temperature)
        return list(set(outputs))


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.ERROR)

    model = RugptChitchat()
    model.load('/home/inkoziev/polygon/chatbot/tmp/rugpt_chitchat')
    #model.load('/home/inkoziev/corpora/EmbeddingModels/rugpt3small_based_on_gpt2')

    # Отладочное тестирование

    if False:
        # Интерактивная сессия - вводим реплики, модель генерирует ответные реплики, всё это накапливается в истории диалога.
        context = []
        while True:
            if context:
                print('Текущий диалог:'.format(len(context)))
                for i, s in enumerate(context, start=1):
                    print('({})  {}'.format(i, s))

            q = input(':> ').strip()
            if q:
                # Реплику с прикрепленной в конце предпосылкой разобьем на 2 части.
                m = re.match(r'^(.+) \[(.+)\]$', q)
                if m:
                    text = m.group(1).strip()
                    premise = m.group(2).strip()
                    context.append(text)
                    context.append('[' + premise + ']')
                else:
                    context.append(q)

                px = model.generate_chitchat(context, num_return_sequences=5)
                print('Сгенерированные варианты ответа:')
                for i, p in enumerate(px):
                    print('[{}]  {}'.format(i, p))
                print('')
                context.append(px[0])
            else:
                # Пустая реплика - значит надо начать новый диалог
                context = []
    elif False:
        # ENRICHED CHITCHAT
        # Интерактивная сессия с вводом вопроса и релевантной предпосылки для тестирования PQA-сценария.
        while True:
            q = input('question:> ').strip()
            p = input('premise:>  ').strip()
            context = [q, '['+p+'.]']
            px = model.generate_chitchat(context, num_return_sequences=5)
            print('Сгенерированные варианты ответа:')
            for i, p in enumerate(px):
                print('[{}]  {}'.format(i, p))
            print('')
    elif False:
        context = ['Привет, Вика!']
        px = model.generate_autoquestions(context, num_return_sequences=5)
        print('Сгенерированные варианты автовопроса:')
        for i, p in enumerate(px):
            print('[{}]  {}'.format(i, p))
        print('')
    elif False:
        context = ['{interpretation}', 'Какую музыку предпочитаешь?', 'Энергичную и мощную']
        px = model.generate_interpretations(context, num_return_sequences=5)
        print('Сгенерированные варианты интерпретации:')
        for i, p in enumerate(px):
            print('[{}]  {}'.format(i, p))
        print('')
    elif True:
        context = ['В какой стране живет Владимир Глуховский?']
        px = model.generate_confabulations(context, num_return_sequences=5)
        print('Сгенерированные конфабуляции:')
        for i, p in enumerate(px):
            print('[{}]  {}'.format(i, p))
        print('')
    elif False:
        while True:
            q = input('Question:> ').strip()
            px = model.generate_confabulations([q], num_return_sequences=5)
            print('Сгенерированные конфабуляции:')
            for i, p in enumerate(px):
                print('[{}]  {}'.format(i, p))
            print('')
    elif False:
        # НАЧАЛО ОТЛАДКИ
        prompt_text = '<s>{chitchat}\n' + '- Как тебя зовут?\n[уклониться от ответа.]\n'
        px = model.generate_output_from_prompt(prompt_text, 5)
        for i, p in enumerate(px):
            print('[{}]  {}'.format(i, p))
        print('')
        # КОНЕЦ ОТЛАДКИ
    else:
        #context = ['[приветствие. сейчас вечер.]']
        context = ['Как ты относишься к украинцам?', '[уклониться от ответа.]']
        px = model.generate_chitchat(context, num_return_sequences=10)
        print('Сгенерированные реплики диалога:')
        for i, p in enumerate(px):
            print('[{}]  {}'.format(i, p))
        print('')
