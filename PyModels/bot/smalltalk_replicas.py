# -*- coding: utf-8 -*-

class SmalltalkReplicas(object):
    def __init__(self, query):
        self.query = query
        self.answers = []

    def add_answer(self, answer):
        self.answers.append(answer)
