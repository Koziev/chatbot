# -*- coding: utf-8 -*-

import unittest
from dummy_answering_machine import DummyAnsweringMachine


class TestDummyAnsweringMachine(unittest.TestCase):
    def setUp(self):
        self.bot = DummyAnsweringMachine()

    def test_echo(self):
        input_phrase = u'входная фраза'
        self.bot.push_phrase('test', input_phrase)
        answer = self.bot.pop_phrase('test')
        self.assertEqual(input_phrase, answer)


if __name__ == '__main__':
    unittest.main()
