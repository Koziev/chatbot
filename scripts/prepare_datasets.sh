set -e

PYTHONPATH=~/polygon/paraphrasing/PyModels
export PYTHONPATH

cd ../PyModels/preparation

python prepare_relevancy_dataset.py

python prepare_relevancy3_dataset.py

python prepare_qa_dataset.py

python prepare_synonymy_dataset.py

python prepare_wordchar_dataset.py

python prepare_word2lemmas.py

python prepare_word_embeddings.py

python prepare_person_changer.py
