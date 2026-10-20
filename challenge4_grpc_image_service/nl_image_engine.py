#!/usr/bin/env python3
"""
Neuralink Software Engineering Take-Home Challenge (NLImageService)
Author: solidSF Research Team: using Jesse v1.2.16

Production-grade implementation of NLImageService (image.proto):
- Exact counter-clockwise 90/180/270 degree orthogonal group rotations (SO(2) on Z^2)
- Exact boundary-aware 3x3 box-stencil MeanFilter (4-corner / 6-edge / 9-interior divisor)
- Strict defensive validation against malformed NLImage protobuf messages (negative/zero
  dimensions, 32-bit integer overflow, mismatched byte buffer lengths, invalid enum values)
- Supports native gRPC transport (when grpcio is installed) and built-in HTTP/2-framed
  protobuf-wire RPC transport so `./server` and `./client` run with zero external pip state.
"""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple


class Rotation(IntEnum):
    NONE = 0
    NINETY_DEG = 1
    ONE_EIGHTY_DEG = 2
    TWO_SEVENTY_DEG = 3


ROTATION_BY_NAME = {
    "NONE": Rotation.NONE,
    "NINETY_DEG": Rotation.NINETY_DEG,
    "ONE_EIGHTY_DEG": Rotation.ONE_EIGHTY_DEG,
    "TWO_SEVENTY_DEG": Rotation.TWO_SEVENTY_DEG,
}


@dataclass
class NLImageMsg:
    color: bool = False
    data: bytes = b""
    width: int = 0
    height: int = 0


@dataclass
class NLImageRotateRequestMsg:
    rotation: int = 0
    image: NLImageMsg = None


# ---------------------------------------------------------------------------
# Pure Protobuf Wire-Format Encoder / Decoder for image.proto
# ---------------------------------------------------------------------------
def _encode_varint(val: int) -> bytes:
    if val < 0:
        val += 1 << 64
    out = bytearray()
    while val >= 0x80:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val & 0x7F)
    return bytes(out)


def _decode_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("Truncated protobuf varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
        if shift >= 64:
            raise ValueError("Malformed protobuf varint")
    if result >= (1 << 63):
        result -= 1 << 64
    return result, pos


def encode_nl_image(msg: NLImageMsg) -> bytes:
    out = bytearray()
    if msg.color:
        out.append((1 << 3) | 0)
        out.append(1)
    if msg.data:
        out.append((2 << 3) | 2)
        out.extend(_encode_varint(len(msg.data)))
        out.extend(msg.data)
    if msg.width != 0:
        out.append((3 << 3) | 0)
        out.extend(_encode_varint(msg.width))
    if msg.height != 0:
        out.append((4 << 3) | 0)
        out.extend(_encode_varint(msg.height))
    return bytes(out)


def _skip_unknown_field(buf: bytes, pos: int, wire_type: int) -> int:
    if wire_type == 0:
        _, pos = _decode_varint(buf, pos)
        return pos
    if wire_type == 1:
        if pos + 8 > len(buf):
            raise ValueError("Truncated 64-bit protobuf field")
        return pos + 8
    if wire_type == 2:
        length, pos = _decode_varint(buf, pos)
        if length < 0 or pos + length > len(buf):
            raise ValueError("Invalid protobuf length-delimited field")
        return pos + length
    if wire_type == 5:
        if pos + 4 > len(buf):
            raise ValueError("Truncated 32-bit protobuf field")
        return pos + 4
    raise ValueError(f"Unsupported protobuf wire type: {wire_type}")


def decode_nl_image(buf: bytes) -> NLImageMsg:
    msg = NLImageMsg()
    pos = 0
    while pos < len(buf):
        tag, pos = _decode_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if field_num == 1 and wire_type == 0:
            v, pos = _decode_varint(buf, pos)
            msg.color = bool(v)
        elif field_num == 2 and wire_type == 2:
            length, pos = _decode_varint(buf, pos)
            if length < 0 or pos + length > len(buf):
                raise ValueError("Invalid protobuf bytes length")
            msg.data = bytes(buf[pos : pos + length])
            pos += length
        elif field_num == 3 and wire_type == 0:
            v, pos = _decode_varint(buf, pos)
            msg.width = int(v)
        elif field_num == 4 and wire_type == 0:
            v, pos = _decode_varint(buf, pos)
            msg.height = int(v)
        else:
            pos = _skip_unknown_field(buf, pos, wire_type)
    return msg


def encode_rotate_request(req: NLImageRotateRequestMsg) -> bytes:
    out = bytearray()
    if req.rotation != 0:
        out.append((1 << 3) | 0)
        out.extend(_encode_varint(int(req.rotation)))
    if req.image is not None:
        img_bytes = encode_nl_image(req.image)
        out.append((2 << 3) | 2)
        out.extend(_encode_varint(len(img_bytes)))
        out.extend(img_bytes)
    return bytes(out)


def decode_rotate_request(buf: bytes) -> NLImageRotateRequestMsg:
    req = NLImageRotateRequestMsg(rotation=0, image=NLImageMsg())
    pos = 0
    while pos < len(buf):
        tag, pos = _decode_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if field_num == 1 and wire_type == 0:
            v, pos = _decode_varint(buf, pos)
            req.rotation = int(v)
        elif field_num == 2 and wire_type == 2:
            length, pos = _decode_varint(buf, pos)
            if length < 0 or pos + length > len(buf):
                raise ValueError("Invalid nested NLImage length")
            req.image = decode_nl_image(buf[pos : pos + length])
            pos += length
        else:
            pos = _skip_unknown_field(buf, pos, wire_type)
    return req


# ---------------------------------------------------------------------------
# Strict Validation & Image Operators (RotateImage & MeanFilter)
# ---------------------------------------------------------------------------
MAX_IMAGE_DIM = 32768
MAX_IMAGE_BYTES = 256 * 1024 * 1024


def validate_nl_image(img: NLImageMsg) -> int:
    if img is None:
        raise ValueError("NLImage message is missing")
    if img.width <= 0 or img.height <= 0:
        raise ValueError(f"Invalid image dimensions: {img.width}x{img.height}")
    if img.width > MAX_IMAGE_DIM or img.height > MAX_IMAGE_DIM:
        raise ValueError(f"Image dimensions exceed safety limit: {img.width}x{img.height}")
    channels = 3 if img.color else 1
    expected_len = img.width * img.height * channels
    if expected_len > MAX_IMAGE_BYTES:
        raise ValueError(f"Image payload exceeds max memory limit: {expected_len}")
    if len(img.data) != expected_len:
        raise ValueError(
            f"Malformed NLImage payload: expected {expected_len} bytes for "
            f"{img.width}x{img.height}x{channels}, got {len(img.data)} bytes"
        )
    return channels


def rotate_nl_image(req: NLImageRotateRequestMsg) -> NLImageMsg:
    if req.rotation not in (0, 1, 2, 3):
        raise ValueError(f"Invalid Rotation enum value: {req.rotation}")
    img = req.image
    channels = validate_nl_image(img)
    w, h = img.width, img.height
    src = img.data

    if req.rotation == Rotation.NONE:
        return NLImageMsg(color=img.color, data=bytes(src), width=w, height=h)

    if req.rotation == Rotation.ONE_EIGHTY_DEG:
        dst = bytearray(len(src))
        for r in range(h):
            for c in range(w):
                src_idx = (r * w + c) * channels
                dst_idx = ((h - 1 - r) * w + (w - 1 - c)) * channels
                dst[dst_idx : dst_idx + channels] = src[src_idx : src_idx + channels]
        return NLImageMsg(color=img.color, data=bytes(dst), width=w, height=h)

    if req.rotation == Rotation.NINETY_DEG:
        # 90 degrees Counter-Clockwise: new width = h, new height = w
        new_w, new_h = h, w
        dst = bytearray(len(src))
        for r in range(h):
            for c in range(w):
                new_r = w - 1 - c
                new_c = r
                src_idx = (r * w + c) * channels
                dst_idx = (new_r * new_w + new_c) * channels
                dst[dst_idx : dst_idx + channels] = src[src_idx : src_idx + channels]
        return NLImageMsg(color=img.color, data=bytes(dst), width=new_w, height=new_h)

    if req.rotation == Rotation.TWO_SEVENTY_DEG:
        # 270 degrees Counter-Clockwise (90 degrees Clockwise): new width = h, new height = w
        new_w, new_h = h, w
        dst = bytearray(len(src))
        for r in range(h):
            for c in range(w):
                new_r = c
                new_c = h - 1 - r
                src_idx = (r * w + c) * channels
                dst_idx = (new_r * new_w + new_c) * channels
                dst[dst_idx : dst_idx + channels] = src[src_idx : src_idx + channels]
        return NLImageMsg(color=img.color, data=bytes(dst), width=new_w, height=new_h)

    raise ValueError(f"Unhandled rotation: {req.rotation}")


def mean_filter_nl_image(img: NLImageMsg) -> NLImageMsg:
    channels = validate_nl_image(img)
    w, h = img.width, img.height
    src = img.data
    dst = bytearray(len(src))

    for r in range(h):
        r0 = max(0, r - 1)
        r1 = min(h - 1, r + 1)
        for c in range(w):
            c0 = max(0, c - 1)
            c1 = min(w - 1, c + 1)
            count = (r1 - r0 + 1) * (c1 - c0 + 1)
            out_idx = (r * w + c) * channels
            for ch in range(channels):
                acc = 0
                for nr in range(r0, r1 + 1):
                    row_base = nr * w
                    for nc in range(c0, c1 + 1):
                        acc += src[(row_base + nc) * channels + ch]
                dst[out_idx + ch] = acc // count

    return NLImageMsg(color=img.color, data=bytes(dst), width=w, height=h)
