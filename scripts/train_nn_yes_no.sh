w2v_path=~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=32.bin
wc2v_path=../data/wordchar2vector.dat
input_path=../data/pqa_yes_no.dat

python ../PyModels/nn_yes_no.py --run_mode train --batch_size 150 --tmp ../tmp --wordchar2vector ${wc2v_path} --word2vector ${w2v_path}
