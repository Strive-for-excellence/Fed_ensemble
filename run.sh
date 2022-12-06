  echo                 launch  -- \
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha $1 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 5000 --use_avg_loss 2 --kalman 1 \
        --name cifar10_fedmd_avg_2_alpha_d_$1_kalman_1
                launch  -- \
        python3 main.py --dataset cifar10 --num_classes 10 --epochs 500 --num_users 4 --local_ep 1 --local_bs 100 --train_num 5000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha $1 \
        --col_policy 2 \
        --pub_data cifar10 --pub_data_num 5000 --use_avg_loss 2 --kalman 1 \
        --name cifar10_fedmd_avg_2_alpha_d_$1_kalman_1