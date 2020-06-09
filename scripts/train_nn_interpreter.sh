 
#PYTHONPATH=../pycode KERAS_BACKEND=tensorflow python3 ../pycode/trainers/nn_interpreter_new2.py --run_mode train --wordchar2vector ../tmp/wc2v.kv --word2vector ../tmp/w2v.kv

PYTHONPATH=../pycode KERAS_BACKEND=tensorflow python3 ../pycode/trainers/nn_seq2seq_interpreter.py \
  --run_mode train \
  --tmp_dir ../tmp \
  --data_dir ../data
