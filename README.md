# tpu_experiment
Repository for TPU Experiment

## By Pytorch

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

2. Create VM and TPU

- Create VM which has many cpu cores by UI or CLI. (ex: n1-highmem-96)
    - Many cpus are important for only Pytorch.

- Create TPU by UI or CLI at same zone where VM is created.


```
$ ctpu up --tpu-size=v3-8 --name=resnet-tutorial --preemptible --zone='us-central1-a' --tf-version=pytorch-nightly
```

3. Pull docker image

```
$ dokcer pull gcr.io/tpu-pytorch/xla:nightly
```

4. Set TPU_IP_ADDRESS (TPU internal IP Address)

```
$ export TPU_IP_ADDRESS=hogehoge
```

5. RUN ImageNet training 

- by MultiProcessing

```
$ docker run --rm -it --shm-size 126G \
    -e XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470" \
    -e XLA_USE_BF16=1 \
    -v $PWD/xla/test:/pytorch/xla/test \
    -v $PWD/input/imagenet/:/imagenet_data/ \
    -v $PWD/reports:/reports \
    -v ~/.config/gcloud:/root/.config/gcloud \
    --ipc=host \
    gcr.io/tpu-pytorch/xla:nightly \
    python /pytorch/xla/test/test_train_mp_imagenet.py --model resnet50 --datadir /imagenet_data/ --num_worker 24 --num_cores 8 --logdir ./reports --log_steps 200
```

- by MultiThreading

```
$ docker run --rm -it --shm-size 126G \
    -e XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470" \
    -e XLA_USE_BF16=1 \
    -v $PWD/xla/test:/pytorch/xla/test \
    -v $PWD/input/imagenet/:/imagenet_data/ \
    -v $PWD/reports:/reports \
    -v ~/.config/gcloud:/root/.config/gcloud \
    --ipc=host \
    gcr.io/tpu-pytorch/xla:nightly \
    python /pytorch/xla/test/test_train_imagenet.py --model resnet50 --datadir /imagenet_data/ --num_worker 24 --num_cores 8 --logdir ./reports --log_steps 200
```

## By Tensorflow1.xx

1. 1-2 is same above training by Pytorch.
    - VM dosen't need so many cpus like Pytorch. (ex: n1-standard-8) 

2. Create TFRecord. (Take long time...)

```
$ cd ./tpu/tools/datasets
$ python imagenet_to_gcs.py --raw_data_dir ./input/imagenet/raw_data --project [gcp-project-name] --gcs_output_path gs://hoge
```

3. RUN ImageNet training

```
$ cd ./tpu/official/resnet
$ python resnet_main.py --tpu=resnet-tutorial --data_dir=gs://hoge/train --model_dir=gs://hoge/model --config_file=configs/cloud/v3-8.yaml
```