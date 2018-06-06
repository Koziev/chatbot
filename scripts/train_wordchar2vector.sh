arch='lstm(cnn)'
seed=5678901
batch_size=250
dims=56
python ../PyModels/wordchar2vector.py --train 1 --i ../tmp/known_words.txt --o ../tmp/wordchar2vector.dat --model_dir ../tmp --arch_type $arch --tunable_char_embeddings 0 --char_dims 0 --batch_size ${batch_size} --dims ${dims} --seed ${seed}
