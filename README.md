# ViTAA: Visual-Textual Attributes Alignment in Person Search by Natural Language

We provide the code for reproducing experiment results of ViTAA

- ECCV2020 conference paper: [pdf](https://arxiv.org/pdf/2005.07327v2.pdf).
- If this work is helpful for your research, please cite **ViTAA**

```
@misc{wang2020vitaa,
    title={ViTAA: Visual-Textual Attributes Alignment in Person Search by Natural Language},
    author={Zhe Wang and Zhiyuan Fang and Jun Wang and Yezhou Yang},
    year={2020},
    eprint={2005.07327},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Benchmark
### CUHK-PEDES
|    Method   |     Features     |  R@1  |  R@5  |  R@10  |
|:-----------:|:----------------:|:-----:|:-----:|:------:|
|   GNA-RNN   |      global      | 19.05 |   -   |  53.64 |
|     CMCE    |      global      | 25.94 |   -   |  60.48 |
|   PWM-ATH   |      global      | 27.14 | 49.45 |  61.02 | 
|  Dual Path  |      global      | 44.40 | 66.26 |  75.07 |
|  CMPM+CMPC  |      global      | 49.37 |   -   |  79.27 | 
|     MIA     |   global+region  | 53.10 | 75.00 |  82.90 |
|     GALM    |  global+keypoint | 54.12 | 75.45 |  82.97 |
|  **ViTAA**  | global+attribute | 55.97 | 75.84 |  83.52 |


## Data preparation
1. Download [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) dataset and save it anywhere you like (e.g. ~/datasets/cuhkpedes/).
2. Download text_attribute_graph ([GoogleDrive](https://drive.google.com/file/d/1Sqm3V97hbqK9GxIwshZejJWLARfu5o1s/view?usp=sharing)
/ [BaiduYun(code: vbss)](https://pan.baidu.com/s/1TIX4lbvZmGwbBNHcRyA1ng)) which are the text phrases parsed from the sentences, and save it in (e.g. ~/datasets/cuhkpedes/).
3. Use the provided [Human Parsing Network](https://github.com/Jarr0d/Human-Parsing-Network) to generate the attribute segmentations, and save it in (e.g. ~/datasets/cuhkpedes/).
4. Run the script in [tools/cuhkpedes/convert_to_json]() to generate the json files as annotations.
```shell
python tools/cuhkpedes/convert_to_json.py --datadir ~/datasets/cuhkpedes/ --outdir datasets/cuhkpedes/annotations
```
Your `datasets` directory should look like this:

````
ViTAA
-- configs
-- tools
-- vitaa
-- datasets
   |-- cuhkpedes
   |   |-- annotations
   |   |   |-- test.json
   |   |   |-- train.json
   |   |   |-- val.json
   |   |-- imgs
   |   |   |-- cam_a
   |   |   |-- cam_b
   |   |   |--  ...
   |   |-- segs
   |   |   |-- cam_a
   |   |   |-- cam_b
   |   |   |--  ...
````


## Training
```
# single-gpu training
python tools/train_net.py --config-file configs/cuhkpedes/bilstm_r50_seg.yaml

# multi-gpu training
We provide the code for distributed training but they haven't been tested
```
Note: We train ViTAA with `batch_size=64` on one Tesla V100 GPU. If your GPU doesn't support such batch size, please follow the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677) to adjust the configuration.


##  Testing
```shell
# single-gpu testing
python tools/test.py --config-file configs/cuhkpedes/bilstm_r50_seg.yaml --checkpoint-file output/cuhkpedes/...
```


##  Human Parsing Network
We separately provide the code of our [Human Parsing Network](https://github.com/Jarr0d/Human-Parsing-Network) because we think it might be a useful tool for the community.


## Acknowledgement
Our codes is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), great thanks to their work.