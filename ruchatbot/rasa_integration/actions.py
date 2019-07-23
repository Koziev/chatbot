# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionHelloWorld(Action):
    def name(self) -> Text:
        return "action_weather"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            sinopsis_date = next(tracker.get_latest_entity_values(u'когда'), None)
            dispatcher.utter_message(u"Прогноз погоды на дату: {}".format(sinopsis_date))
        except:
            dispatcher.utter_message(u"Прогноз погоды - слот даты не найден")

        return []


import logging
import ruchatbot

class ActionQA(Action):
    def __init__(self):
        try:
            profile_path = '/home/inkoziev/polygon/chatbot/data/profile_1.json'
            models_folder = '/home/inkoziev/polygon/chatbot/tmp'
            data_folder = '/home/inkoziev/polygon/chatbot/data'
            w2v_folder = '/home/inkoziev/polygon/chatbot/tmp'
            self.bot = ruchatbot.create_qa_bot(profile_path, models_folder, data_folder, w2v_folder, debugging=True)
            self.user_id = "test_rasa"
        except Exception as ex:
            logging.error(ex)

    def name(self) -> Text:
        return "action_qa"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            question = tracker.latest_message['text']
            logging.debug(u'Pushing question="%s"', question)
            self.bot.push_phrase(self.user_id, question)
            answer = self.bot.pop_phrase(self.user_id)
            logging.debug(u'answer="%s"', answer)
            dispatcher.utter_message(answer)
        except:
            dispatcher.utter_message(u"Возникла ошибка при формировании ответа во внешнем боте")

        return []
