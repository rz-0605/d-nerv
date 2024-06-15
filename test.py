from dataset import *
import torchvision.transforms as transforms
import argparse
import torch

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.clip_size = 8
args.keyframe_quality = 3
args.batchSize = 32
args.workers = 1

ds = Dataset_DNeRV_UVG(
    args, 
    transforms.Compose([transforms.ToTensor()]), 
    transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4519, 0.4505, 0.4519], [0.2805, 0.2747, 0.2787])])
)

train_dataloader = torch.utils.data.DataLoader(ds, batch_size=args.batchSize, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, 
        # collate_fn=my_collate_fn
)

for i, (video, norm_idx, keyframe, backward_distance, frame_mask, name) in enumerate(train_dataloader):
    print(name, len(name))
    print(video.shape)