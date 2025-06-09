import logging
import os
from typing import List

import torch

from gradient_mechanics.data import video_demuxing
from gradient_mechanics.data import video_transforms

logger = logging.getLogger(__name__)


class VideoClipDataset(torch.utils.data.Dataset):
    def __init__(self, index_frame, group_num, num_cameras=7, on_demand=False):
        """
        Initialize the VideoDataset.

        Args:
            video_file_path: Path to the video file.
            episode_length: Number of frames in an episode.
            episode_stride: Stride between episode frames.
        """
        self._index_frame = index_frame
        self._num_cameras = num_cameras
        self._group_num = group_num
        self._on_demand = on_demand
        self._indexing_demuxer = None
        if self._on_demand:
            self._indexing_demuxer = video_demuxing.IndexingDemuxerOndemand([""]*num_cameras, num_cameras, group_num)
        
    def __len__(self):
        """Return the total number of episodes."""
        frame_count = 0
        for video_dir ,clip_info in self.index_frame.items():
            for clip_id , frame_count in clip_info.items():
                frame_count += frame_count

        length = frame_count // self.group_num
        return length

    def __getitem__(self, index: int) -> List[video_transforms.PacketBuffers]:
        """Return the episode at the given index."""

        episode_buffers = []
        if self._on_demand:
            # file_paths = []
            # frame_idxs = []
            # for clip_info in index:
            #     clip_path = clip_info[0]
            #     video_paths = [os.path.join(clip_path,f) for f in os.listdir(clip_path) if f.endswith('.mp4')]
            #     file_paths = file_paths + video_paths
            #     frame_idxs = frame_idxs + [clip_info[1]]*len(video_paths)
            #     print(f"clip_info: {clip_info}")
            #     print(f"file_paths: {file_paths}")
            #     print(f"frame_idxs: {frame_idxs}")
            # self._indexing_demuxer.update_path(file_paths)
            # episode_buffers = self._indexing_demuxer.packet_buffers_for_frame_idx(frame_idxs)
            for clip_info in index:
                clip_path = clip_info[0]
                video_paths = [os.path.join(clip_path,f) for f in os.listdir(clip_path) if f.endswith('.mp4')]
                frame_idx = clip_info[1]
                self._indexing_demuxer.update_path(video_paths)
                episode_buffers.append(self._indexing_demuxer.packet_buffers_for_frame_idx(frame_idx))
        else:
            for clip_info in index:
                clip_path = clip_info[0]
                video_paths = [os.path.join(clip_path,f) for f in os.listdir(clip_path) if f.endswith('.mp4')]
                frame_idx = [clip_info[1]]* len(video_paths)
                for video_path, frame_id in zip(video_paths, frame_idx):
                    indexing_demuxer = video_demuxing.IndexingDemuxer(video_path)
                    episode_buffers.append(indexing_demuxer.packet_buffers_for_frame_idx(frame_id))

        return episode_buffers
