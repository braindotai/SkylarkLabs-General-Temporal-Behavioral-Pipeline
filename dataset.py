import torch
from torch import nn
import dxeon as dx
from moviepy.editor import VideoFileClip
from torchvision import io
from torchvision.datasets import DatasetFolder
from math import ceil, floor

DATASET_ROOT = dx.os.path.join('DATASET', 'UCF-101')
VIDEO_PATHS = dx.glob(dx.os.path.join(DATASET_ROOT, '*', '*'))
TEMPORAL_DIMENSION = 16
SPATIAL_DIMENSION = [192, 256]

video_reader = io.VideoReader(VIDEO_PATHS[0])
for frame1 in video_reader:
    pass

# transforms = torch.nn.Sequential(
#     dx.transforms.CenterCrop(10),
#     dx.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# )
# scripted_transforms = torch.jit.script(transforms)
# print(scripted_transforms(torch.ones(3, 128, 128)).shape, torch.__version__)

class VideoAugmentation(nn.Module):
    def __init__(
        self,
        mean = [0.43216, 0.394666, 0.37645],
        std = [0.22803, 0.22145, 0.216989]
    ):
        super().__init__()

        self.mean = mean
        self.std = std
    
    def forward(self, batch):
        frames = []

        if self.training and torch.rand(()) > 0.5:
            batch = dx.TF.hflip(batch)

        batch = dx.TF.convert_image_dtype(batch, torch.float32)
        batch = torch.stack([dx.TF.normalize(frame, self.mean, self.std, inplace = True) for frame in batch])

        return batch.permute(0, 2, 1, 3, 4)

def loader(video_path):
    video_reader = io.VideoReader(video_path)
    metadata = video_reader.get_metadata()

    fps = metadata['video']['fps'][0]
    fps = floor(fps) if fps % 1 <= 0.5 else ceil(fps)
    
    duration = metadata['video']['duration'][0]
    sampling_rate = fps * duration / TEMPORAL_DIMENSION
    sampling_rate = floor(sampling_rate) if sampling_rate % 1 <= 0.5 else ceil(sampling_rate)
    
    sampled_frames = []

    for idx, frame in enumerate(video_reader, 1):
        frame = dx.TF.resize(frame['data'], SPATIAL_DIMENSION)
        
        if idx % sampling_rate == 0:
            sampled_frames.append(frame)
        
        if idx == 1:
            first_frame = frame
    
    if len(sampled_frames) == TEMPORAL_DIMENSION - 1:
        sampled_frames.insert(0, first_frame)
    elif len(sampled_frames) == TEMPORAL_DIMENSION + 1:
        sampled_frames.pop(0)
    elif len(sampled_frames) > TEMPORAL_DIMENSION:
        sampled_frames = sampled_frames[:TEMPORAL_DIMENSION]
    elif len(sampled_frames) < TEMPORAL_DIMENSION:
        sampled_frames.extend([sampled_frames[-1]] * (TEMPORAL_DIMENSION - len(sampled_frames)))
    
    return torch.stack(sampled_frames)

dataset = DatasetFolder(
    root = DATASET_ROOT,
    loader = loader,
    extensions = ('mp4', 'avi')
)

dataloader = dx.DataLoader(dataset, batch_size = 32, shuffle = True, pin_memory = True, num_workers = 8)

for x, y in dataloader:
    break

dx.stats.summarize(x)

# aug = VideoAugmentation().cuda()
# x = x.cuda()
# dx.utils.benchmark_performance(aug, x, runs = 100)

aug = VideoAugmentation().cpu()
x = x.cpu()
dx.utils.benchmark_performance(aug, x, runs = 100)

aug = torch.jit.script(VideoAugmentation().eval()).cuda()
x = x.cuda()
dx.utils.benchmark_performance(aug, x, runs = 100)

for video, label in zip(aug(x), y):
    dx.io.gif.write(video.permute(1, 0, 2, 3).cpu(), output_path = f'samples/{label}.gif')
