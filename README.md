# tf_image_template
A template for tensorflow image handling project.

### File structure

```yaml
.
├── checkpoints
│   └── sample_0001.ckpt
├── config.py
├── data_loader
│   ├── datadb.py
│   ├── data_loader.py
│   ├── imdb.py
│   ├── pipeline.py
│   └── transform.py
├── models
│   ├── base_model.py
│   ├── layers.py
│   ├── losses.py
│   ├── mynet.py
│   ├── resnet.py
│   └── vgg.py
├── README.md
├── scripts
│   └── test_batch.py
├── main.py
└── utils
    └── misc_utils.py
```

### Train your own network
```shell script
    python main.py --train --input_dir train/ --output_dir checkpoint/ --epochs 100 --which_direction BtoA [--resume]
```

### Test the model
```shell script
    python main.py --test --input_dir val/ --output_dir test_results/ --checkpoint checkpoint/ 
```

### Visulization
```shell script
    tensorboard --logdir=checkpoint
```

--train --input_dir c --val_dir val --output_dir checkpoint --epochs 200 --which_direction BtoA --eva_freq 250 --checkpoint checkpoint/model-2499