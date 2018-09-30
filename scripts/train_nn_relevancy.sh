w2v_path=~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=32.bin
wc2v_path=../data/wordchar2vector.dat

KERAS_BACKEND=tensorflow python ../PyModels/nn_relevancy.py --run_mode train --batch_size 150 --arch 'cnn2' --classifier 'merge2' --input ../data/premise_question_relevancy.csv --tmp ../tmp --wordchar2vector ${wc2v_path} --word2vector ${w2v_path}

