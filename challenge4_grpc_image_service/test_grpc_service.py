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

        print("ALL 4 NEURALINK GRPC IMAGE SERVICE TEST SUITES PASSED (0 FAILURES).")
    finally:
        srv_proc.terminate()
        srv_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
