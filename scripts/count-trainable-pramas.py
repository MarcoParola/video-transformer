import os
import numpy as np
import torch
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import hydra
from tqdm import tqdm

from src.models import SetCriterion
from src.datasets import collateFunction, COCODataset
from src.utils import load_model
from src.utils.misc import cast2Float
from src.utils.utils import load_weights
from src.utils.boxOps import boxCxcywh2Xyxy, gIoU, boxIoU
from src.models.matcher import HungarianMatcher


@hydra.main(config_path="../config", config_name="config")
def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    os.makedirs(args.outputDir, exist_ok=True)

    # load data and model
    #test_dataset = COCODataset(args.dataDir, args.testAnnFile, args.numClass)

    args.sequenceLength = 14

    args.model = 'detr'
    model = load_model(args).to(device) 
    args.model = 'detr-base' 
    print(f"{args.model} Number of parameters: {sum([p.numel() for p in model.parameters()])}")

    args.model = 'early-concat-detr'
    model = load_model(args).to(device)
    args.model = 'early-concat-detr-base'
    print(f"{args.model} Number of parameters: {sum([p.numel() for p in model.parameters()])}")

    args.model = 'yolos'
    model = load_model(args).to(device)
    args.model = 'yolos-tiny'
    print(f"{args.model} Number of parameters: {sum([p.numel() for p in model.parameters()])}")

    args.model = 'early-concat-yolos'
    model = load_model(args).to(device)
    args.model = 'early-concat-yolos-tiny'
    print(f"{args.model} Number of parameters: {sum([p.numel() for p in model.parameters()])}")

    args.yolos.backboneName = 'base'
    args.model = 'yolos'
    model = load_model(args).to(device)
    args.model = 'yolos-base'
    print(f"{args.model} Number of parameters: {sum([p.numel() for p in model.parameters()])}")

    args.model = 'early-mul-detr'
    model = load_model(args).to(device)
    args.model = 'early-sum-detr-base'
    print(f"{args.model} Number of parameters: {sum([p.numel() for p in model.parameters()])}")

    
    

    
    
    
   
if __name__ == '__main__':
    main()