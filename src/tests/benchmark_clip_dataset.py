import argparse
import json
import logging
import os
import time
import random

from torch.utils.data import Sampler
from gradient_mechanics.data import torch_loading, torchdata_loading
from gradient_mechanics.data import transforms
from gradient_mechanics.data import video_transforms
from tests import clip_dataset

class StreamingVideoClipSampler(Sampler):
    def __init__(self, index_frame, group_num):
        self.index_frame = index_frame
        self.group_num = group_num
        self.clip_ids = list(index_frame.keys())
        self._generate_groups()

    def _generate_groups(self):
        """divide the clip index into groups"""
        self.groups_lst = []
        for video_dir ,clip_info in self.index_frame.items():
            for clip_id , frame_count in clip_info.items():
                clip_path = os.path.join(video_dir, clip_id)
                for i in range(frame_count):
                    self.groups_lst.append((clip_path, i))

        self.groups = []
        self.group_size = len(self.groups_lst) // self.group_num
        for i in range(0, self.group_size * self.group_num, self.group_size):
            group = self.groups_lst[i:i + self.group_size]
            self.groups.append(group)

    def __iter__(self):
        #Return clip index for each batch
        for i in range(0, self.group_size):
            batch = []
            for group in self.groups:
                batch.append(group[i])
            yield batch

    def __len__(self):
        #Return total number of batches
        return self.group_size

class RandomAccessVideoClipSampler(Sampler):
    def __init__(self, index_frame, group_num):
        self.index_frame = index_frame
        self.group_num = group_num
        self.clip_ids = list(index_frame.keys())
        self._generate_groups()

    def _generate_groups(self):
        """divide the clip index into groups"""
        self.groups_lst = []
        for video_dir ,clip_info in self.index_frame.items():
            for clip_id , frame_count in clip_info.items():
                clip_path = os.path.join(video_dir, clip_id)
                for i in range(frame_count):
                    self.groups_lst.append((clip_path, i))
        random.shuffle(self.groups_lst)
        self.groups = []
        self.group_size = len(self.groups_lst) // self.group_num
        for i in range(0, self.group_size * self.group_num, self.group_size):
            group = self.groups_lst[i:i + self.group_size]
            self.groups.append(group)

    def __iter__(self):
        #Return clip index for each batch
        for i in range(0, self.group_size):
            batch = []
            for group in self.groups:
                batch.append(group[i])
            yield batch

    def __len__(self):
        #Return total number of batches
        return self.group_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Benchmark ClipDataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataloader-cls", type=str, default="torch", choices=["torch", "torchdata"]
    )
    parser.add_argument(
        "--sampler", type=str, default="streaming", choices=["streaming", "randomaccess"]
    )
    parser.add_argument(
        "--codec",
        type=lambda codec: video_transforms.Codec[codec],
        default="HEVC",
        choices=list(video_transforms.Codec),
    )
    parser.add_argument("--index_file", type=str, help='Path to the index_frame JSON file (default: %(default)s)',default= '/data/ndas/index_frame.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--group_num", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Load index_frame from JSON file
    with open(args.index_file, 'r') as f:
        index_frame = json.load(f)

    if args.dataloader_cls == "torchdata":
        dataloader_cls = torchdata_loading.GPUDataLoader
    elif args.dataloader_cls == "torch":
        dataloader_cls = torch_loading.GPUDataLoader
    else:
        raise ValueError(f"Invalid dataloader class: {args.dataloader_cls}")
    codec = args.codec
    batch_size = args.batch_size
    num_workers = args.num_workers
    device_id = args.device_id

    construction_started_at = time.perf_counter()
    dataset = clip_dataset.VideoClipDataset(index_frame=index_frame, group_num=args.group_num)
    sampler = StreamingVideoClipSampler(index_frame=index_frame, group_num=args.group_num)
    if args.sampler == "streaming":
        sampler = StreamingVideoClipSampler(index_frame=index_frame, group_num=args.group_num)
    elif args.sampler == "randomaccess":
        sampler = RandomAccessVideoClipSampler(index_frame=index_frame, group_num=args.group_num)
    else:
        raise ValueError(f"Invalid sampler type: {args.sampler}")

    gpu_transforms = [
        video_transforms.DecodeVideo(device_id=device_id, codec=codec),
        transforms.ToTensor(device_id=device_id),
    ]
    loader = torch_loading.GPUDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        # GPU Specific
        gpu_device=device_id,
        gpu_prefetch_factor=1,
        gpu_transforms=gpu_transforms,
    )
    construction_ended_at = time.perf_counter()
    started_at = time.perf_counter()
    samples_loaded = 0
    batches_loaded = 0
    load_gaps = []

    load_started_at = time.perf_counter()
    first_batch_received_at = None
    for i, batch in enumerate(loader):
        load_ended_at = time.perf_counter()
        load_gaps.append(load_ended_at - load_started_at)
        if first_batch_received_at is None:
            first_batch_received_at = load_ended_at

        load_started_at = time.perf_counter()
        print(f"Batch: {i} - {len(batch)} frames in {load_gaps[-1]:.4f} seconds")
        batches_loaded += 1
        for sample in batch:
            samples_loaded += sample.shape[0]

    ended_at = time.perf_counter()

    throughput = samples_loaded / (ended_at - started_at)
    print(f"Samples loaded: {samples_loaded}")
    print(
        f"Construction time: {construction_ended_at - construction_started_at:.2f} seconds"
    )
    print(f"Time to first batch: {first_batch_received_at - started_at:.2f} seconds")
    print(f"Time taken: {ended_at - started_at:.2f} seconds")
    print(f"Throughput: {throughput:.2f} samples per second")
    print(
        f"Throughput: {batches_loaded / (ended_at - started_at):.2f} batches per second"
    )

    warmup = 2
    load_gaps = load_gaps[warmup:]
    print(f"Mean load gap: {sum(load_gaps) / len(load_gaps):.4f}")
    print(f"Max load gap: {max(load_gaps):.4f}")
    print(f"Min load gap: {min(load_gaps):.4f}")
    print(f"Num workers: {num_workers}")
    # print out the dataloader class and module
    print(f"Dataloader class: {dataloader_cls.__name__}")
    print(f"Dataloader module: {dataloader_cls.__module__}")
