import os
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data.dataset import Dataset
import src.utils.transforms as T
import hydra


class COCODataset(Dataset):
    def __init__(self, 
            root: str, 
            annotation: str, 
            numClass: int = 4, 
            removeBackground: bool = True,):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.numClass = numClass
        self.removeBackground = removeBackground
        self.transforms = T.Compose([
            T.ToTensor()
        ])

        self.newIndex = {}
        classes = []
        for i, (k, v) in enumerate(self.coco.cats.items()):
            self.newIndex[k] = i
            classes.append(v['name'])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        imgID = self.ids[idx]
        imgInfo = self.coco.imgs[imgID]        
        imgPath = os.path.join(self.root, imgInfo['file_name'])
        image = Image.open(imgPath).convert('L')

        # crop the pil image by removing the top part, remove the first 96 pixels
        if self.removeBackground:
            image = image.crop((0, 96, image.size[0], image.size[1]))

        annotations = self.loadAnnotations(imgID, imgWidth=image.size[0], imgHeight=image.size[1])

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64)-1,}
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64)-1,}

        image, targets = self.transforms(image, targets)

        return image, targets


    def loadAnnotations(self, imgID: int, imgWidth: int, imgHeight: int) -> np.ndarray:
        """ load the annotations (bbox and label) for the image with the given ID
        Args:
            imgID: image ID
            imgWidth: image width
            imgHeight: image height
        Returns:
            np.ndarray: annotations for the image
        """
        ans = []
        for annotation in self.coco.imgToAnns[imgID]:
            cat = self.newIndex[annotation['category_id']]
            bbox = annotation['bbox']

            # if you want to crop the background (top part of the image), NOTE: you need to remove a multiple of 16 pixels
            if self.removeBackground:
                if bbox[1] + bbox[3] < 96:
                    continue
                  
                bbox[1] -= 96
                if bbox[1] < 0:
                    bbox[3] -= 96 - bbox[1]
                    bbox[1] = 0
                    if bbox[3] < 5:
                        continue

            bbox = [val / imgHeight if i % 2 else val / imgWidth for i, val in enumerate(bbox)]
            ans.append(bbox + [cat])

        return np.asarray(ans)


def collateFunction(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, Tuple[Dict[str, Tensor]]]:
    """ collate function for the dataloader to transform the list of tuples into a tuple of tensors
    Args:
        batch: list of tuples of the form (image, target)
    Returns:
        tuple of tensors
    """
    batch = tuple(zip(*batch))
    return torch.stack(batch[0]), batch[1]



# TEST MAIN
@hydra.main(config_path='../../config', config_name='config', version_base='1.1')
def main(args):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torch.utils.data import DataLoader

    num_classes = 4
    data_folder = 'data'
    data_file = 'data/new/Test.json'
    data_folder = os.path.join(args.currentDir, data_folder)
    data_file = os.path.join(args.currentDir, data_file)
    dataset = COCODataset(data_folder, data_file, num_classes, removeBackground=args.cropBackground)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collateFunction)
    print(dataset.__len__())
    img, trg = dataset.__getitem__(0)
    print(img.shape, trg['boxes'].shape, trg['labels'].shape)

    
    '''
    for i in range(50):
        img, target = dataset.__getitem__(i)
        print(img.shape, target['boxes'].shape, target['labels'].shape)
    
    # test for checking the size of the metadata sequence
    sizes = [0] * 100
    from tqdm import tqdm
    for j in tqdm(range(dataset.__len__())):
        image, meta, target = dataset.__getitem__(j)
        print(image.shape, meta.shape, target['boxes'].shape, target['labels'].shape)
    print(sizes)
    
    testing_dir = 'outputs/test/'
    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir)

    for j in range(20):
        image, target = dataset.__getitem__(j)
        fig, ax = plt.subplots()
        ax.imshow(image.permute(1, 2, 0))

        # add title using the image name
        imgID = dataset.ids[j]
        imgInfo = dataset.coco.imgs[imgID]
        ax.set_title(imgInfo['file_name'])

        print(len(target['boxes']))

        for i in range(len(target['boxes'])):
            img_w, img_h = image.size(2), image.size(1)
            x, y, w, h = target['boxes'][i]
            x, y = x*img_w, y*img_h 
            w, h = w*img_w, h*img_h
            rect = patches.Rectangle((x,y), w,h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.savefig(f'{testing_dir}img_{j}.png')
        plt.close(fig)
    '''
    

if __name__ == '__main__':
    main()