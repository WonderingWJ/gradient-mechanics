import argparse
import logging
import time
import os
import torch
import numpy as np
from torch.nn.functional import cosine_similarity
from torchvision.utils import save_image

from gradient_mechanics.data import torch_loading, torchdata_loading
from gradient_mechanics.data import transforms
from gradient_mechanics.data import video_transforms
from tests import video_dataset
import pdb

def compare_batches(current_batch, ref_batch):
    """
    Compare two batches and return various distance metrics.
    
    Args:
        current_batch: Current batch tensor
        ref_batch: Reference batch tensor
        
    Returns:
        dict: Dictionary containing comparison metrics
    """
    metrics = {}
    
    # Convert both tensors to float type
    current_batch = current_batch.float()
    ref_batch = ref_batch.float()
    
    # Compare shapes
    metrics['shapes_match'] = current_batch.shape == ref_batch.shape
    metrics['current_shape'] = current_batch.shape
    metrics['ref_shape'] = ref_batch.shape
    
    if not metrics['shapes_match']:
        return metrics
    
    # Calculate absolute differences
    diff = torch.abs(current_batch - ref_batch)
    metrics['max_diff'] = torch.max(diff).item()
    metrics['mean_diff'] = torch.mean(diff).item()
    
    # Calculate L2 distance
    l2_dist = torch.norm(current_batch - ref_batch, p=2).item()
    metrics['l2_distance'] = l2_dist
    
    # Calculate cosine distance (1 - cosine similarity)
    # Reshape tensors to 2D for cosine similarity calculation
    current_flat = current_batch.reshape(current_batch.shape[0], -1)
    ref_flat = ref_batch.reshape(ref_batch.shape[0], -1)
    cosine_sim = cosine_similarity(current_flat, ref_flat, dim=1)
    metrics['cosine_distance'] = (1 - cosine_sim).mean().item()
    
    return metrics

def print_comparison_metrics(metrics):
    """Print comparison metrics in a formatted way."""
    print("\nComparison Results:")
    print(f"Shapes match: {metrics['shapes_match']}")
    if not metrics['shapes_match']:
        print(f"Current shape: {metrics['current_shape']}")
        print(f"Reference shape: {metrics['ref_shape']}")
        return
    
    print(f"Max difference: {metrics['max_diff']:.6f}")
    print(f"Mean difference: {metrics['mean_diff']:.6f}")
    print(f"L2 distance: {metrics['l2_distance']:.6f}")
    print(f"Cosine distance: {metrics['cosine_distance']:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Benchmark VideoDataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video_file_path", type=str)
    parser.add_argument(
        "--dataloader-cls", type=str, default="torch", choices=["torch", "torchdata"]
    )
    parser.add_argument(
        "--codec",
        type=lambda codec: video_transforms.Codec[codec],
        default="HEVC",
        choices=list(video_transforms.Codec),
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--episode-length", type=int, default=1)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--use-check", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    video_file_path = args.video_file_path
    if args.dataloader_cls == "torchdata":
        print("torchdata")
        dataloader_cls = torchdata_loading.GPUDataLoader
    elif args.dataloader_cls == "torch":
        print("torch")
        dataloader_cls = torch_loading.GPUDataLoader
    else:
        raise ValueError(f"Invalid dataloader class: {args.dataloader_cls}")
    codec = args.codec
    shuffle = args.shuffle
    episode_length = args.episode_length
    episode_stride = args.episode_stride
    batch_size = args.batch_size
    num_workers = args.num_workers
    device_id = args.device_id

    construction_started_at = time.perf_counter()
    dataset = video_dataset.VideoDataset(
        video_file_path, episode_length=episode_length, episode_stride=episode_stride, on_demand=True
    )

    gpu_transforms = [
        video_transforms.DecodeVideoOnDemand(device_id=device_id, codec=codec, num_cameras=1, num_group=1),
        # transforms.ToTensor(device_id=device_id),
    ]
    loader = torch_loading.GPUDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
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
        if args.use_check == 1 and i > 100:
            break
        load_ended_at = time.perf_counter()
        load_gaps.append(load_ended_at - load_started_at)
        if first_batch_received_at is None:
            first_batch_received_at = load_ended_at

        load_started_at = time.perf_counter()
        print(f"Batch: {i} - {len(batch)} frames in {load_gaps[-1]:.4f} seconds")
        batches_loaded += 1
        
        # Save batch data in binary format
        # os.makedirs('./ref', exist_ok=True)
        # torch.save(batch, f'./ref/batch_{i}.pt')
        
        # Load reference data and compare
        if args.use_check == 1:
            ref_path = f'./ref_benchmark_video_dataset/batch_{i}.pt'
            if os.path.exists(ref_path):
                ref_batch = torch.load(ref_path)
                ref_batch = ref_batch.unsqueeze(0)  # Add dimension of size 1 at the highest dimension
            print("ref_batch: ", ref_batch.shape)
            print("batch: ", batch.shape)
            batch = batch.permute(0, 1, 4, 2, 3)  # Transpose dimensions to match ref_batch shape
            
            # Save images, ignoring the highest two dimensions
            os.makedirs('./comparison_images', exist_ok=True)
            # Remove the highest two dimensions (both are 1)
            batch_img = batch.squeeze(0).squeeze(0)  # Shape: [3, 1080, 1920]
            ref_batch_img = ref_batch.squeeze(0).squeeze(0)  # Shape: [3, 1080, 1920]
            
            # Convert to float and ensure values are in [0, 1] range
            batch_img = batch_img.float() / 255.0
            ref_batch_img = ref_batch_img.float() / 255.0
            
            # save_image(batch_img, f'./comparison_images/batch_{i}.png')
            # save_image(ref_batch_img, f'./comparison_images/ref_batch_{i}.png')
            
            metrics = compare_batches(batch, ref_batch)
            print_comparison_metrics(metrics)
        
        for sample in batch:
            print("batch type:", type(batch), batch.shape)
            print("sample type:", type(sample), sample.shape)
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
    print(f"Shuffle: {shuffle}")
