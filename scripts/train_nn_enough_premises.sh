w2v_path=~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin
#wc2v_path=../data/wordchar2vector.dat

KERAS_BACKEND=tensorflow python ../PyModels/nn_enough_premises.py --run_mode train --batch_size 250 --word2vector ${w2v_path}
