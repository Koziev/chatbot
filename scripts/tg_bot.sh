PYTHONPATH=.. \
python3 ../ruchatbot/bot/conversation_engine.py \
--mode telegram \
--chatbot_dir .. \
--log ../tmp/axioma.log \
--profile ../data/profile_1.json \
--bert ../tmp/rubert-tiny \
--db ../tmp/facts_db.sqlite
