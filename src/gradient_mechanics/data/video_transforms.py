import ctypes
import enum
from typing import List
import typing

import numpy as np
import PyNvVideoCodec as nvc
import cvcuda

from gradient_mechanics.data import transforms

import torch
from torch.utils.data._utils import collate

def packet_from_buffer(buffer: torch.ByteTensor) -> nvc.PacketData:
    packet = nvc.PacketData()
    packet.bsl_data = buffer.data_ptr()
    packet.bsl = buffer.numel()
    return packet

def buffer_from_packet(packet: nvc.PacketData) -> torch.ByteTensor:
    numpy_byte_array = np.ctypeslib.as_array(
        ctypes.cast(packet.bsl_data, ctypes.POINTER(ctypes.c_uint8)),
        shape=(packet.bsl,),
    )
    return torch.from_numpy(numpy_byte_array.copy())

def _get_tensor_hwc(height: int, width: int, device: int):
    return cvcuda.as_tensor(
        torch.zeros((1, height, width, 3), dtype=torch.uint8, device=f"cuda:{device}"),
        layout="NHWC",
    )

def _get_tensor_hwc_y8(height: int, width: int, device: int):
    return cvcuda.as_tensor(
        torch.zeros((1, height, width, 1), dtype=torch.uint8, device=f"cuda:{device}"),
        layout="NHWC",
    )

class PacketBuffersBatch(typing.NamedTuple):
    """Batch of PacketBuffers."""

    samples: List["PacketBuffers"]


class PacketBuffers(typing.NamedTuple):
    """Frames targeted for decoding."""
    packets: list[torch.Tensor]
    """List of packets. This contains packets for the target frames and any frames the target frames depend on."""

    @classmethod
    def collate(
        cls, samples: List["PacketBuffers"], *, collate_fn_map=None
    ) -> PacketBuffersBatch:
        """"""
        return PacketBuffersBatch(samples=samples)


collate.default_collate_fn_map[PacketBuffers] = PacketBuffers.collate

class Codec(enum.Enum):
    """Codec for video decoding."""

    AV1 = nvc.cudaVideoCodec.AV1
    H264 = nvc.cudaVideoCodec.H264
    HEVC = nvc.cudaVideoCodec.HEVC


class DecodeVideo(transforms.Transform):
    def __init__(self, *, codec: Codec = Codec.H264, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_input_type(PacketBuffersBatch)
        self._codec = codec
        self._decoder = nvc.CreateDecoder(
            gpuid=self.device_id,
            codec=codec.value,
            cudacontext=0,
            cudastream=0,
            usedevicememory=True,
        )
        self.height = 1080
        self.width = 1920

    def to_rgb_tensor(self, nvcv_image: cvcuda.Image):
        nvcv_tensor: cvcuda.Tensor = cvcuda.as_tensor(nvcv_image)
        nvcv_hwc = _get_tensor_hwc_y8(
           nvcv_tensor.shape[2], nvcv_tensor.shape[3], self.device_id
        )
        nvcv_rgb = _get_tensor_hwc(self.height, self.width, self.device_id)
        cvcuda.reformat_into(nvcv_hwc, nvcv_tensor)    
        cvcuda.advcvtcolor_into(
            nvcv_rgb,
            nvcv_hwc,
            cvcuda.ColorConversion.YUV2RGB_NV12,
            cvcuda.ColorSpec.BT601,
        )
    
        return nvcv_rgb
    def transform(self, batch: List[PacketBuffersBatch]) -> List[cvcuda.Tensor]:
        decoded_batches: list[cvcuda.Tensor] = []
        for item in batch:
            decoded_batch = self.decode_batch(item)
            decoded_batches.append(decoded_batch)

        return decoded_batches

    def decode_batch(self, batch: PacketBuffersBatch) -> list[cvcuda.Tensor]:
        decoded_batch: list[cvcuda.Tensor] = []
        for packet_buffers in batch.samples:
            decoded = self.decode_sample(packet_buffers)
            decoded_batch.append(cvcuda.stack(decoded))
        return decoded_batch

    def decode_sample(
        self, episode_packet_buffer: PacketBuffers
    ) -> list[cvcuda.Tensor]:
        """
        Decode the episode from the given EpisodePacketBuffer.

        Args:
            episode_packet_buffer: EpisodePacketBuffer containing the packets and frames to decode.

        Returns:
            List of decoded frames.
        """
        target_tensor: List[cvcuda.Tensor] = []
        # Decoder buffers packets so we need to keep track of the decoded frames
        # packet offset to be able to map the decoded frames to the target frames.
        frame_packet_offset = 0
        total_packets = len(episode_packet_buffer.packets)
        for packet_offset, packet_buffer in enumerate(episode_packet_buffer.packets):
            packet = packet_from_buffer(packet_buffer)

            # Note: We copy frames to tensors because a DecodedFrame is really an
            # iterator which will point to a different frame after the next call to Decode.
            decoded: nvc.DecodedFrame = self._decoder.Decode(packet)
        
        # Send a null packet to flush the decoder.
        null_packet = nvc.PacketData()
        null_packet.bsl = 0
        null_packet.bsl_data = 0
        decoded = self._decoder.Decode(null_packet)

        nvcv_image: cvcuda.Image = cvcuda.as_image(decoded[-1].nvcv_image(), cvcuda.Format.Y8)
        torch_tensor = self.to_rgb_tensor(nvcv_image)
        target_tensor.append(torch_tensor)
        
        if not all(tensor is not None for tensor in target_tensor):
            raise ValueError(
                f"DecodeVideo failed to decode all packet buffers. Make sure the video is encoded with the selected codec: '{self._codec.name}'"
            )

        return list(target_tensor)
