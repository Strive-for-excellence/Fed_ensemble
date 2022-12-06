
# drop_rate = 0.1
## 狄利克雷non-iid
## CNN
###        0

        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1 --model CNN  \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_CNN_P_0

###        1

        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1 --model CNN \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 1000 --use_avg_loss 1 \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_CNN_P_1

###        2 kalman1

        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1 --model CNN \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 1000 --use_avg_loss 2 --kalman 1 \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_CNN_P_2
       
###        4
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1 --model CNN \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 1000 --use_avg_loss 4 --kalman 1 \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_CNN_P_4

###        5 基于 max {p_c} c \in [1,numclasses]
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1 --model CNN \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 1000 --use_avg_loss 5  \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_CNN_P_5
# DNN
        0

        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1   \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_DNN_P_0

        1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1  \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 1000 --use_avg_loss 1 \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_DNN_P_1

        2
        kalman1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1  \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 1000 --use_avg_loss 2 --kalman 1 \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_DNN_P_2
       
        4
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 1000 --use_avg_loss 4 --kalman 1 \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_DNN_P_4

        5 基于 max {p_c} c \in [1,numclasses]
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 1000 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1  \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 1000 --use_avg_loss 5  \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_DNN_P_5

# pubdata_num = 500

        1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1  \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 500 --use_avg_loss 1 \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_pub_500_DNN_P_1

        2
        kalman1
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1  \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 500 --use_avg_loss 2 --kalman 1 \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_pub_500_DNN_P_2
       
        4
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1 \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 500 --use_avg_loss 4 --kalman 1 \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_pub_500_DNN_P_4

        5 基于 max {p_c} c \in [1,numclasses]
        python3 main.py --dataset mnist --num_classes 10 --epochs 300 --num_users 20 --local_ep 1 --local_bs 1000 --train_num 100 \
        --policy 1   \
        --iid 0 --noniid dirichlet --alpha 0.01 --drop_rate 0.1  \
        --col_policy 2 \
        --pub_data mnist --pub_data_num 500 --use_avg_loss 5  \
        --name mnist_d_alpha_0.1_drop_rate_0.1_model_pub_500_DNN_P_5