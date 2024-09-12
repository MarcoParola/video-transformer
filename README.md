# conditioning-transformer

[![size](https://img.shields.io/github/languages/code-size/MarcoParola/video-transformer?style=plastic)]()
[![license](https://img.shields.io/static/v1?label=OS&message=Linux&color=green&style=plastic)]()
[![Python](https://img.shields.io/static/v1?label=Python&message=3.10&color=blue&style=plastic)]()


Design of a transformer-based architecture for multi-frame aggregation for object detection downstream task:
- DEtection TRanformer (DETR)
- You Only Look at One Sequence (YOLOS)

## **Multiple frame fusion**

We use two distinct architectures for aggregate multiple frames:
- BasicVSR, originally proposed for Video Super Resolution in [BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond](https://arxiv.org/abs/2012.02181). We use the implementation proposed [here](https://github.com/sunny2109/BasicVSR_IconVSR_PyTorch)
- EST-RNN, originally proposed for Video Deblurring in [Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510188.pdf). We use the [official implementation](https://github.com/zzh-tech/ESTRNN) proposed by the authors. We also propose a variation of the aggregation block to remove artifacts during the reconstruction. Set the reconstruction param in the [estrnn config file](./config/estrnn.yaml#L) choosing between `original` or `interpolation`.

## **Installation**

To install the project, simply clone the repository and get the necessary dependencies:
```sh
git clone https://github.com/MarcoParola/conditioning-transformer.git
cd conditioning-transformer
mkdir models data
```

Create and activate virtual environment, then install dependencies. 
```sh
python -m venv env
. env/bin/activate
python -m pip install -r requirements.txt 
```

Next, create a new project on [Weights & Biases](https://wandb.ai/site). Log in and paste your API key when prompted. Then edit your wandb name on the [config file](./config/config.yaml#L81)
```sh
wandb login 
```

## **Usage**

To perform a training run by setting `model` parameter:
```sh
python train.py model=detr
```
`model` can assume the following value `detr`, `yolos`, `vsr-yolos`, `estrnn-yolos`.
TODO extend: estrnn-detr and vsr-detr



## Acknowledgement
Special thanks to [@clive819](https://github.com/clive819) for making an implementation of DETR public [here](https://github.com/clive819/Modified-DETR). Special thanks to [@hustvl](https://github.com/hustvl) for YOLOS [original implementation](https://github.com/hustvl/YOLOS)
