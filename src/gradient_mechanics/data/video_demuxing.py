import logging
import os
from typing import List, Optional

import PyNvVideoCodec as nvc
import PyNvOnDemandDecoder as nvc_ondemand
import torch
from gradient_mechanics.data import video_transforms

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
        self._video_file_paths = video_file_paths
        self._nv_gop_dec = nvc_ondemand.CreateGopDecoder(
            # maxfiles = num_cameras,
            maxfiles = num_cameras * num_group,
            usedevicememory = 1,
            iGpu = 0,
            cachedir="",
        )
        self._packet_buffers = None

    def __len__(self) -> int:
        return 605

    def packet_buffers_for_frame_idx(
        self, frame_idx: int
    ) -> video_transforms.PacketOndemandBuffers:
        """
        Fetch packets and dependencies for the given frame indices.

        Args:
            frame_idx: List of frame indices to fetch packets for.

        Returns:
            PacketOndemandBuffers object containing the target frames, packet frames, and packets.
        """

        try:
            gop_packets = self._nv_gop_dec.GetPackets(self._video_file_paths, [frame_idx]*len(self._video_file_paths))
        except Exception as e:
            logger.error(f"Error fetching packets for frame {frame_idx} with video_file_paths: {self._video_file_paths}. Error: {e}")
            exit(1)

        return video_transforms.PacketOndemandBuffers(
            gop_packets=[gop_packets],
            target_frame_list=[frame_idx]*len(self._video_file_paths),
        )

    def packet_buffers_for_frame_idx_list(
        self, frame_idx_list: List[int]
    ) -> video_transforms.PacketOndemandBuffers:
        """
        Fetch packets and dependencies for the given frame indices.

        Args:
            frame_idx: List of frame indices to fetch packets for.

        Returns:
            PacketOndemandBuffers object containing the target frames, packet frames, and packets.
        """

        gop_packets = None
        use_cache = True

        if self._packet_buffers is None:
            use_cache = False
        else:
            for i in range(len(self._packet_buffers.filepaths)):
                if self._packet_buffers.filepaths[i] != self._video_file_paths[i]:
                    self._packet_buffers = self._packet_buffers
                    use_cache = False
                    break
                if frame_idx_list[i] < self._packet_buffers.first_frame_ids[i] or frame_idx_list[i] >= self._packet_buffers.first_frame_ids[i] + self._packet_buffers.gop_lens[i]:
                    use_cache = False
                    break
        
        if use_cache:
            gop_packets = self._packet_buffers
        else:
            print("new decode !!")
            try:
                gop_packets = self._nv_gop_dec.GetPackets(self._video_file_paths, frame_idx_list)
                self._packet_buffers = gop_packets
            except Exception as e:
                logger.error(f"Error fetching packets for frame {frame_idx_list} with video_file_paths: {self._video_file_paths}. Error: {e}")
                exit(1)

        return video_transforms.PacketOndemandBuffers(
            gop_packets=[gop_packets],
            target_frame_list=frame_idx_list,
        )

    def update_path(self, video_file_paths: List[str]):
        for video_file_path in video_file_paths:
            if not os.path.exists(video_file_path):
                raise FileNotFoundError(f"Video file {video_file_path} does not exist")
        self._video_file_paths = video_file_paths
    