import logging
import os
from typing import List, Optional

import PyNvVideoCodec as nvc
import PyNvOnDemandDecoder as nvc_ondemand
import torch
from gradient_mechanics.data import video_transforms
import torch.cuda.nvtx as nvtx

logger = logging.getLogger(__name__)


class IndexingDemuxer:
    def __init__(self, video_file_path: str) -> None:
        """
        Initialize the IndexingDemuxer.

        Args:
            video_file_path: Path to the video file to demux.
            video_index: Video index to use for demuxing.
        """
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file {video_file_path} does not exist")
        self._video_file_path = video_file_path
        self._nv_dmx = nvc.CreateDemuxer(filename=video_file_path)

    def __len__(self) -> int:
        return 605

    def packet_buffers_for_frame_idx(
        self, frame_idx: int
    ) -> video_transforms.PacketBuffers:
        """
        Fetch packets and dependencies for the given frame indices.

        Args:
            frame_idx: List of frame indices to fetch packets for.

        Returns:
            PacketBuffers object containing the target frames, packet frames, and packets.
        """

        #if not (0 <= frame_idx < len(self)):
        #    raise ValueError("frame_idx must be within the range of the video")

        packets = self._fetch_packets(frame_idx)

        return video_transforms.PacketBuffers(
            packets=packets,
        )

    def _fetch_packets(self, frame_idx: int) -> List[torch.Tensor]:
        """
        Fetch packets for the given packet indices.

        Args:
            frame_idx: frame idx to fetch for its decode.
        """
        seek_idx = int(frame_idx / 30) * 30
        packet_index_to_packet = {}
        self._nv_dmx.Seek(seek_idx)
        for packet_idx, packet_iter in enumerate(self._nv_dmx):
            # Stop after we've hit the frame we want to actually load.
            if packet_idx > 30 or seek_idx + packet_idx > frame_idx:
                break
            if packet_iter.bsl_data is None or packet_iter.bsl == 0:
                logger.info("Skipping packet with no bsl_data")
                continue
            copied_bsl = video_transforms.buffer_from_packet(packet_iter)

            packet_index_to_packet[packet_idx+seek_idx] = copied_bsl

        return [packet for packet in packet_index_to_packet.values()]

class IndexingDemuxerOndemand:
    def __init__(self, video_file_paths: List[str], num_cameras :int, num_group :int) -> None:
        """
        Initialize the IndexingDemuxer.

        Args:
            video_file_path: Path to the video file to demux.
        """
        print(f"IndexingDemuxerOndemand init !")
        self._num_group = num_group
        self._num_cameras = num_cameras
        self._video_file_paths = video_file_paths
        self._nv_gop_dec = nvc_ondemand.CreateGopDecoder(
            maxfiles = num_cameras,
            usedevicememory = 1,
            iGpu = 0,
        )
        self._packet_buffers = [None] * num_group

    def __len__(self) -> int:
        return 605

    def check_use_cache(self, frame_idx_list: List[int], group_idx: int) -> bool:
        use_cache = True

        if self._packet_buffers[group_idx] is None:
            use_cache = False
        else:
            for i in range(len(self._packet_buffers[group_idx].filepaths)):
                if self._packet_buffers[group_idx].filepaths[i] != self._video_file_paths[i]:
                    print("diff path!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    use_cache = False
                    break
                if frame_idx_list[i] < self._packet_buffers[group_idx].first_frame_ids[i] or frame_idx_list[i] >= self._packet_buffers[group_idx].first_frame_ids[i] + self._packet_buffers[group_idx].gop_lens[i]:
                    use_cache = False
                    break
        
        return use_cache

    def packet_buffers_for_frame_idx(
        self, frame_idx: int
    ) -> video_transforms.PacketOndemandBuffers:
        """
        Fetch packets and dependencies for the given frame indices.

        Args:
            frame_idx: frame indice to fetch packets for.

        Returns:
            PacketOndemandBuffers object containing the target frames, packet frames, and packets.
        """

        gop_packets = None
        result_gop_packets = []

        use_cache = self.check_use_cache([frame_idx], 0)

        if use_cache == False:
            try:
                gop_packets = self._nv_gop_dec.GetPackets(self._video_file_paths, [frame_idx]*len(self._video_file_paths), False)
                self._packet_buffers[0] = gop_packets
            except Exception as e:
                logger.error(f"Error fetching packets for frame {frame_idx} with video_file_paths: {self._video_file_paths}. Error: {e}")
                exit(1)
            
            result_gop_packets = [gop_packets]
        
        return video_transforms.PacketOndemandBuffers(
            gop_packets=result_gop_packets,
            target_frame_list=[frame_idx]*len(self._video_file_paths),
            use_cache=use_cache,
            group_idx=0,
        )

    def packet_buffers_for_frame_idx_list(
        self, frame_idx_list: List[int], group_idx: int = 0
    ) -> video_transforms.PacketOndemandBuffers:
        """
        Fetch packets and dependencies for the given frame indices.

        Args:
            frame_idx: List of frame indices to fetch packets for.

        Returns:
            PacketOndemandBuffers object containing the target frames, packet frames, and packets.
        """
        nvtx.range_push("packet_buffers_for_frame_idx_list")

        if group_idx >= self._num_group:
            raise ValueError(f"Group index {group_idx} is out of range. The number of groups is {self._num_group}")

        gop_packets = None
        tensors = None

        use_cache = self.check_use_cache(frame_idx_list, group_idx)

        if use_cache == False:
            print(f"use_cache: {use_cache}, group_idx: {group_idx}")
            if (self._packet_buffers[group_idx] is None):
                print(f"cached: {None}")
            else:
                print(f"cached: {self._packet_buffers[group_idx].first_frame_ids}")
            print(f"frame_idx_list: {frame_idx_list}")

        
        if use_cache == False:
            try:
                gop_packets = self._nv_gop_dec.GetPackets(self._video_file_paths, frame_idx_list, False)
                self._packet_buffers[group_idx] = gop_packets
            except Exception as e:
                logger.error(f"Error fetching packets for frame {frame_idx_list} with video_file_paths: {self._video_file_paths}. Error: {e}")
                exit(1)
            
            tensors = convert_lists_to_tensors(gop_packets.packet_binary_data)

        result = video_transforms.PacketOndemandBuffers(
            gop_packets=gop_packets,
            target_frame_list=frame_idx_list,
            use_cache=use_cache,
            tensors=tensors,
            group_idx=group_idx,
        )
        nvtx.range_pop()  # packet_buffers_for_frame_idx_list
        return result

    def update_path(self, video_file_paths: List[str]):
        for video_file_path in video_file_paths:
            if not os.path.exists(video_file_path):
                raise FileNotFoundError(f"Video file {video_file_path} does not exist")
        self._video_file_paths = video_file_paths
    
def convert_lists_to_tensors(src_list):
    result = []
    for src in src_list:
        tensor = torch.tensor(src, dtype=torch.uint8)
        result.append(tensor)
    return result
