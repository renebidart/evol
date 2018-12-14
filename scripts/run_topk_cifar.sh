# smaller test ones:
# add in a topk after each block to normal resnet18
python evolve-topk.py --files_df_loc /media/rene/data/learn-lr/CIFAR10/files_df.pkl --SAVE_PATH /media/rene/data/learn-lr/CIFAR10/models  --batch_size 512 --device cuda:1 --dataset CIFAR10 --epochs 5 --MAX_GENERATIONS 3 --NPOPULATION 3 --within_block_act relu --after_block_act topk
# swap all activations for topk, no relu
python evolve-topk.py --files_df_loc /media/rene/data/learn-lr/CIFAR10/files_df.pkl --SAVE_PATH /media/rene/data/learn-lr/CIFAR10/models  --batch_size 512 --device cuda:1 --dataset CIFAR10 --epochs 5 --MAX_GENERATIONS 3 --NPOPULATION 3 --within_block_act topk --after_block_act None

# REAL TESTS
# add in a topk after each block to normal resnet18
python evolve-topk.py --files_df_loc /media/rene/data/learn-lr/CIFAR10/files_df.pkl --SAVE_PATH /media/rene/data/learn-lr/CIFAR10/models  --batch_size 512 --device cuda:1 --dataset CIFAR10 --epochs 25 --MAX_GENERATIONS 30 --NPOPULATION 15 --within_block_act relu --after_block_act topk
# swap all activations for topk, no relu
python evolve-topk.py --files_df_loc /media/rene/data/learn-lr/CIFAR10/files_df.pkl --SAVE_PATH /media/rene/data/learn-lr/CIFAR10/models  --batch_size 512 --device cuda:1 --dataset CIFAR10 --epochs 25 --MAX_GENERATIONS 30 --NPOPULATION 15 --within_block_act topk --after_block_act None

