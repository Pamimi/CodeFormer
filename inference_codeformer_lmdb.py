#!/usr/bin/env python3
"""对 LMDB 中的图像批量运行 CodeFormer；默认只处理前若干张。

与 inference_codeformer.py 共用同一套模型加载与单张人脸推理逻辑。LMDB 内为已对齐人脸时，
按 has_aligned 路径：resize 到 512 后直接送入网络，跳过整图检测与 align/crop。
"""
import argparse
import io
import os
import sys
from pathlib import Path
from warnings import warn

import cv2
import lmdb
import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import normalize

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.misc import is_gray

# 保证从任意工作目录运行时能加载 basicsr / facelib（与 inference_codeformer.py 一致）
_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

pretrain_model_url = {
    "restoration": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
}


class LMDBEngine:
    """与 third_party/GFPGAN/inference_gfpgan_lmdb.py 中相同的 LMDB 读写封装。"""

    def __init__(self, lmdb_path, write=False):
        self._write = write
        self._manual_close = False
        self._lmdb_path = lmdb_path
        if write and not os.path.exists(lmdb_path):
            os.makedirs(lmdb_path)
        if write:
            self._lmdb_env = lmdb.open(lmdb_path, map_size=1099511627776)
            self._lmdb_txn = self._lmdb_env.begin(write=True)
        else:
            self._lmdb_env = lmdb.open(
                lmdb_path, readonly=True, lock=False, readahead=False, meminit=True
            )
            self._lmdb_txn = self._lmdb_env.begin(write=False)

    def __getitem__(self, key_name):
        payload = self._lmdb_txn.get(key_name.encode())
        if payload is None:
            raise KeyError("Key:{} Not Found!".format(key_name))
        try:
            image_buf = torch.tensor(np.frombuffer(payload, dtype=np.uint8))
            data = torchvision.io.decode_image(image_buf, mode=torchvision.io.ImageReadMode.RGB)
        except Exception:
            data = torch.load(io.BytesIO(payload), weights_only=True)
        return data

    def __del__(self):
        if not self._manual_close:
            warn("Writing engine not mannuly closed!", RuntimeWarning)
            self.close()

    def keys(self):
        all_keys = list(self._lmdb_txn.cursor().iternext(values=False))
        return [key.decode() for key in all_keys]

    def close(self):
        if self._write:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)
        self._lmdb_env.close()
        self._manual_close = True


def _tensor_chw_rgb_to_bgr_uint8(img_t: torch.Tensor) -> np.ndarray:
    """LMDB decode 为 RGB uint8 CHW -> OpenCV BGR HWC uint8。"""
    if img_t.dim() != 3:
        raise ValueError(f"expect CHW image tensor, got shape {tuple(img_t.shape)}")
    rgb = img_t.detach().cpu()
    if rgb.dtype != torch.uint8:
        rgb = rgb.clamp(0, 255).to(torch.uint8)
    rgb_hwc = rgb.permute(1, 2, 0).numpy()
    return cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)


def _restore_one_face(net, device, img_bgr: np.ndarray, w: float) -> np.ndarray:
    """单张 BGR  uint8 人脸图（建议已对齐），返回 BGR uint8 还原结果。"""
    img = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    cropped_face_t = img2tensor(img / 255.0, bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            output = net(cropped_face_t, w=w, adain=True)[0]
            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as error:
        print(f"\tFailed inference for CodeFormer: {error}")
        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
    return restored_face.astype("uint8")


def main():
    parser = argparse.ArgumentParser(
        description="CodeFormer inference on LMDB images (aligned faces, no face det/crop)."
    )
    parser.add_argument("lmdb_path", type=str, help="LMDB 目录路径")
    parser.add_argument("-o", "--output", type=str, default="results_codeformer_lmdb", help="输出目录")
    parser.add_argument(
        "-w",
        "--fidelity_weight",
        type=float,
        default=0.5,
        help="Balance quality and fidelity (同 inference_codeformer -w). Default: 0.5",
    )
    parser.add_argument("-n", "--max-images", type=int, default=10, help="最多推理条数（默认 10）")
    parser.add_argument("--ext", type=str, default="png", help="输出扩展名: png | jpg")
    parser.add_argument(
        "--save-restored-only",
        action="store_true",
        help="仅保存还原图（restored_imgs），不写左右对比（否则写入 cmp/）",
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output, "restored_imgs"), exist_ok=True)
    if not args.save_restored_only:
        os.makedirs(os.path.join(args.output, "cmp"), exist_ok=True)

    device = get_device()
    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    ckpt_path = load_file_from_url(
        url=pretrain_model_url["restoration"],
        model_dir="weights/CodeFormer",
        progress=True,
        file_name=None,
    )
    checkpoint = torch.load(ckpt_path, map_location=device)["params_ema"]
    net.load_state_dict(checkpoint)
    net.eval()

    engine = LMDBEngine(args.lmdb_path, write=False)
    try:
        all_keys = engine.keys()
        keys = all_keys[: args.max_images]
        for key in keys:
            print(f"Processing {key} ...")
            img_t = engine[key]
            input_bgr = _tensor_chw_rgb_to_bgr_uint8(img_t)
            if is_gray(input_bgr, threshold=10):
                print("\tGrayscale-like input: True")

            restored_face = _restore_one_face(net, device, input_bgr, args.fidelity_weight)
            safe_key = key.replace("/", "_")
            out_name = f"{safe_key}.{args.ext}"

            if not args.save_restored_only:
                inp512 = cv2.resize(input_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
                cmp_img = np.concatenate((inp512, restored_face), axis=1)
                imwrite(cmp_img, os.path.join(args.output, "cmp", out_name))
            imwrite(restored_face, os.path.join(args.output, "restored_imgs", out_name))
    finally:
        engine.close()

    if args.save_restored_only:
        print(f"Done. Restored images under [{os.path.join(args.output, 'restored_imgs')}]")
    else:
        print(
            f"Done. Comparison (输入512|还原) under [{os.path.join(args.output, 'cmp')}], "
            f"restored under [{os.path.join(args.output, 'restored_imgs')}]"
        )


if __name__ == "__main__":
    main()
