echo "start..."
python3 train.py --source i --target n --k 5 --n_samples 4 --msc_coeff 2.0 --multi_gpu 0 --pre_train 2000 --ila_switch_iter 800 --mu 80
