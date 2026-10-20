#!/usr/bin/env python3
"""
End-to-End Integration & Adversarial Protobuf Verification Suite for Challenge 4
(NLImageService: RotateImage + MeanFilter)
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image

import urllib.error
import urllib.request
from nl_image_engine import (
    NLImageMsg,
    NLImageRotateRequestMsg,
    Rotation,
    decode_nl_image,
    encode_nl_image,
    encode_rotate_request,
)


def call_rpc(host: str, port: int, method: str, payload: bytes) -> bytes:
    url = f"http://{host}:{port}/NLImageService/{method}"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/x-protobuf"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"RPC {method} failed ({err.code}): {detail}") from err


def main():
    base = Path(__file__).resolve().parent
    port = 50559
    host = "127.0.0.1"

    srv_proc = subprocess.Popen(
        [sys.executable, str(base / "server"), "--host", host, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.4)

    try:
        print("=== Neuralink Software Engineering Take-Home (NLImageService) Suite ===")
        # 1. Verify exact 3x3 MeanFilter example from image.proto docstring:
        #   A B C      10 20 30
        #   D E F  =   40 50 60
        #   G H I      70 80 90
        data_3x3 = bytes([10, 20, 30, 40, 50, 60, 70, 80, 90])
        msg_3x3 = NLImageMsg(color=False, data=data_3x3, width=3, height=3)
        resp = decode_nl_image(call_rpc(host, port, "MeanFilter", encode_nl_image(msg_3x3)))
        assert resp.width == 3 and resp.height == 3 and not resp.color
        # Corner A = (10+20+40+50)//4 = 30
        # Edge D   = (40+10+20+50+70+80)//6 = 45
        # Center E = (10+20+30+40+50+60+70+80+90)//9 = 50
        assert resp.data[0] == 30, f"Expected A=30, got {resp.data[0]}"
        assert resp.data[3] == 45, f"Expected D=45, got {resp.data[3]}"
        assert resp.data[4] == 50, f"Expected E=50, got {resp.data[4]}"
        print("[PASS] Test 1: Exact 3x3 Grayscale MeanFilter Stencil (A=30, D=45, E=50)")

        # 2. Verify 90/180/270 CCW rotations on non-square 3x2 RGB image
        rgb_3x2 = bytes([
            1, 2, 3,    4, 5, 6,    7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18
        ])
        msg_rgb = NLImageMsg(color=True, data=rgb_3x2, width=3, height=2)
        rot90 = decode_nl_image(
            call_rpc(
                host,
                port,
                "RotateImage",
                encode_rotate_request(
                    NLImageRotateRequestMsg(rotation=Rotation.NINETY_DEG, image=msg_rgb)
                ),
            )
        )
        assert rot90.width == 2 and rot90.height == 3 and rot90.color
        # Top-right pixel (7,8,9) rotated 90 CCW becomes top-left (0,0)
        assert rot90.data[0:3] == bytes([7, 8, 9])
        # Four consecutive 90-degree rotations return exact identity
        cur = msg_rgb
        for _ in range(4):
            cur = decode_nl_image(
                call_rpc(
                    host,
                    port,
                    "RotateImage",
                    encode_rotate_request(
                        NLImageRotateRequestMsg(rotation=Rotation.NINETY_DEG, image=cur)
                    ),
                )
            )
        assert cur.width == 3 and cur.height == 2 and cur.data == rgb_3x2
        print("[PASS] Test 2: Non-Square RGB 90/180/270 CCW Rotation & SO(2) 4-Fold Identity")

        # 3. Verify CLI `./client` with simultaneous `--rotate NINETY_DEG` and `--mean` on PNG
        in_png = base / "test_in.png"
        out_png = base / "test_out.png"
        Image.frombytes("RGB", (3, 2), rgb_3x2).save(in_png)
        subprocess.run(
            [
                sys.executable,
                str(base / "client"),
                "--host",
                host,
                "--port",
                str(port),
                "--input",
                str(in_png),
                "--output",
                str(out_png),
                "--rotate",
                "NINETY_DEG",
                "--mean",
            ],
            check=True,
        )
        with Image.open(out_png) as out_im:
            assert out_im.size == (2, 3)
        in_png.unlink(missing_ok=True)
        out_png.unlink(missing_ok=True)
        print("[PASS] Test 3: End-to-End `./client` PNG I/O with Simultaneous `--rotate` & `--mean`")

        # 4. Verify Malformed NLImage Protobuf Payload Rejection (per prompt.md specification)
        bad_img = NLImageMsg(color=True, data=b"short_corrupt_bytes", width=100, height=100)
        rejected = False
        try:
            call_rpc(host, port, "MeanFilter", encode_nl_image(bad_img))
        except RuntimeError:
            rejected = True
        assert rejected, "Server failed to reject malformed NLImage payload!"
        print("[PASS] Test 4: Malformed NLImage Protobuf Payload Gracefully Rejected")

        # 5. Verify Rectangular Non-Square (5x3 RGB and 4x7 Grayscale) 90/180/270 CCW vs PIL.Image.Transpose
        rect_w, rect_h = 5, 3
        rect_rgb_bytes = bytes(range(1, rect_w * rect_h * 3 + 1))
        pil_rgb = Image.frombytes("RGB", (rect_w, rect_h), rect_rgb_bytes)
        msg_rect_rgb = NLImageMsg(color=True, data=rect_rgb_bytes, width=rect_w, height=rect_h)
        for rot_enum, pil_op, exp_size in [
            (Rotation.NINETY_DEG, Image.Transpose.ROTATE_90, (rect_h, rect_w)),
            (Rotation.ONE_EIGHTY_DEG, Image.Transpose.ROTATE_180, (rect_w, rect_h)),
            (Rotation.TWO_SEVENTY_DEG, Image.Transpose.ROTATE_270, (rect_h, rect_w)),
        ]:
            rot_resp = decode_nl_image(
                call_rpc(
                    host,
                    port,
                    "RotateImage",
                    encode_rotate_request(NLImageRotateRequestMsg(rotation=rot_enum, image=msg_rect_rgb)),
                )
            )
            pil_rot = pil_rgb.transpose(pil_op)
            assert (rot_resp.width, rot_resp.height) == exp_size
            assert rot_resp.data == pil_rot.tobytes(), f"Mismatch vs PIL for {rot_enum.name}"

        gray_w, gray_h = 4, 7
        gray_bytes = bytes(range(10, 10 + gray_w * gray_h))
        pil_gray = Image.frombytes("L", (gray_w, gray_h), gray_bytes)
        msg_rect_gray = NLImageMsg(color=False, data=gray_bytes, width=gray_w, height=gray_h)
        for rot_enum, pil_op, exp_size in [
            (Rotation.NINETY_DEG, Image.Transpose.ROTATE_90, (gray_h, gray_w)),
            (Rotation.TWO_SEVENTY_DEG, Image.Transpose.ROTATE_270, (gray_h, gray_w)),
        ]:
            rot_resp = decode_nl_image(
                call_rpc(
                    host,
                    port,
                    "RotateImage",
                    encode_rotate_request(NLImageRotateRequestMsg(rotation=rot_enum, image=msg_rect_gray)),
                )
            )
            pil_rot = pil_gray.transpose(pil_op)
            assert (rot_resp.width, rot_resp.height) == exp_size
            assert rot_resp.data == pil_rot.tobytes()
        print("[PASS] Test 5: Rectangular (5x3 RGB & 4x7 Grayscale) 90/180/270 CCW Exact Parity vs PIL")

        # 6. Verify MeanFilter on 1x1, 1x5, 5x1, and 3-channel RGB (zero cross-channel bleed & zero uint8 overflow)
        msg_1x1 = NLImageMsg(color=False, data=bytes([255]), width=1, height=1)
        resp_1x1 = decode_nl_image(call_rpc(host, port, "MeanFilter", encode_nl_image(msg_1x1)))
        assert resp_1x1.width == 1 and resp_1x1.height == 1 and resp_1x1.data == bytes([255])

        row_1x5 = bytes([10, 20, 30, 40, 50])
        msg_1x5 = NLImageMsg(color=False, data=row_1x5, width=5, height=1)
        resp_1x5 = decode_nl_image(call_rpc(host, port, "MeanFilter", encode_nl_image(msg_1x5)))
        assert resp_1x5.width == 5 and resp_1x5.height == 1
        assert list(resp_1x5.data) == [15, 20, 30, 40, 45]

        msg_5x1 = NLImageMsg(color=False, data=row_1x5, width=1, height=5)
        resp_5x1 = decode_nl_image(call_rpc(host, port, "MeanFilter", encode_nl_image(msg_5x1)))
        assert resp_5x1.width == 1 and resp_5x1.height == 5
        assert list(resp_5x1.data) == [15, 20, 30, 40, 45]

        # 3-channel RGB: R=255 everywhere, G=0 everywhere, B=100 everywhere -> zero cross-channel bleed
        rgb_iso = bytes([255, 0, 100] * 12)
        msg_rgb_iso = NLImageMsg(color=True, data=rgb_iso, width=4, height=3)
        resp_rgb_iso = decode_nl_image(call_rpc(host, port, "MeanFilter", encode_nl_image(msg_rgb_iso)))
        assert resp_rgb_iso.data == rgb_iso
        print("[PASS] Test 6: MeanFilter 1x1, 1x5, 5x1 & 3-Channel RGB Isolation / Overflow Invariance")

        print("ALL 6 NEURALINK GRPC IMAGE SERVICE TEST SUITES PASSED (0 FAILURES).")
    finally:
        srv_proc.terminate()
        srv_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
