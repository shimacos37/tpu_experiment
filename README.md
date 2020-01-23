# tpu_experiment
Repository for TPU Experiment

## By Pytorch

1. Pull docker image

```
$ dokcer pull gcr.io/tpu-pytorch/xla:r0.5
```

1. Download Imagenet data and save different directory per label. (like below)

```
├── train
│   ├── n01440764
│   ├── n01443537
│   ├── n01484850
│   ├── n01491361
│   ├── n01494475
|   :
|
└── val
    ├── n01440764
    ├── n01443537
    ├── n01484850
    ├── n01491361
    ├── n01494475
    :
```

1. Start VM and TPU

- Create VM which has many cpu cores by UI or CLI. (ex: n1-highmem-96)
    - Many cpus are important for only Pytorch

- Create TPU by UI or CLI at zone where VM is started

```
$ ctpu up --tpu-size=v3-8 --name=resnet-tutorial --preemptible --zone='us-central1-a'
```

1. Set TPU_IP_ADDRESS (TPU internal IP Address)

```
$ export TPU_IP_ADDRESS=hogehoge
```

1. RUN ImageNet training 

- by MultiProcessing

```
$ (VM) xwxwdocker run --rm -it --shm-size 126G \
    -e XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470" \
    -e XLA_USE_BF16=1 \
    -v $PWD/xla/test:/pytorch/xla/test \
    -v $PWD/input/imagenet/:/imagenet_data/ \
    -v $PWD/reports:/reports \
    -v ~/.config/gcloud:/root/.config/gcloud \
    --ipc=host \
    gcr.io/tpu-pytorch/xla:r0.5 \
    python /pytorch/xla/test/test_train_mp_imagenet.py --model resnet50 --datadir /imagenet_data/ --num_worker 24 --num_cores 8 --logdir ./reports --log_steps 200
```

- by MultiThreading

```
$ (VM) docker run --rm -it --shm-size 126G \
    -e XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470" \
    -e XLA_USE_BF16=1 \
    -v $PWD/xla/test:/pytorch/xla/test \
    -v $PWD/input/imagenet/:/imagenet_data/ \
    -v $PWD/reports:/reports \
    -v ~/.config/gcloud:/root/.config/gcloud \
    --ipc=host \
    gcr.io/tpu-pytorch/xla:r0.5 \
    python /pytorch/xla/test/test_train_imagenet.py --model resnet50 --datadir /imagenet_data/ --num_worker 24 --num_cores 8 --logdir ./reports --log_steps 200
```
