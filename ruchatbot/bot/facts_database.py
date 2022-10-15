"""
Черновая реализация интерфейса хранилища фактов в БД.

15.10.2022 Первая реализация на SQLite, стараемся делать максимально абстрактный API
           для потенциального перехода на другие СУБД
"""

import sqlite3


class FactsDatabase(object):
    def __init__(self, connection_string):
        self.con = sqlite3.connect(connection_string, check_same_thread=False)

        # проверим, есть ли нужные таблицы в БД
        cur = self.con.cursor()
        res = cur.execute("SELECT name FROM sqlite_master WHERE name='chatbot_facts'")
        if res.fetchone() is None:
            self.create_database()

    def create_database(self):
        cur = self.con.cursor()
        cur.execute("""CREATE TABLE chatbot_facts(id INTEGER PRIMARY KEY AUTOINCREMENT,
                       interlocutor_id TEXT NOT NULL,
                       fact_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                       fact_text TEXT UNIQUE NOT NULL,
                       fact_tag TEXT)""")

    def store_fact(self, interlocutor_id, fact_text, fact_tag):
        cur = self.con.cursor()
        cur.execute("INSERT INTO chatbot_facts(interlocutor_id, fact_text, fact_tag) VALUES (?, ?, ?)", (interlocutor_id, fact_text, fact_tag))
        self.con.commit()

    def reset_facts(self, interlocutor_id):
        cur = self.con.cursor()
        cur.execute("DELETE FROM chatbot_facts WHERE interlocutor_id=?", (interlocutor_id,))
        self.con.commit()

    def load_facts(self, interlocutor_id):
        cur = self.con.cursor()
        res = cur.execute("SELECT fact_text, fact_tag FROM chatbot_facts WHERE interlocutor_id=?", (interlocutor_id,))
        facts = cur.fetchall()
        return facts

    def update_tagged_fact(self, interlocutor_id, fact_text, fact_tag):
        cur = self.con.cursor()
        res = cur.execute("SELECT id FROM chatbot_facts WHERE interlocutor_id=? AND fact_tag=?", (interlocutor_id, fact_tag))
        fact_id = cur.fetchone()
        if fact_id is None:
            res = cur.execute("INSERT INTO chatbot_facts(fact_text, interlocutor_id, fact_tag) VALUES(?, ?, ?)", (fact_text, interlocutor_id, fact_tag))
        else:
            res = cur.execute("UPDATE chatbot_facts set fact_text=? WHERE id=?", (fact_text, fact_id[0]))
        self.con.commit()

    def find_tagged_fact(self, interlocutor_id, fact_tag):
        cur = self.con.cursor()
        res = cur.execute("SELECT fact_text FROM chatbot_facts WHERE interlocutor_id=? AND fact_tag=?", (interlocutor_id, fact_tag))
        fact = cur.fetchone()
        return fact
