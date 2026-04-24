#!/usr/bin/env python3
"""对 LMDB 中的图像批量运行 CodeFormer；默认只处理前若干张。

- aligned：LMDB 为已对齐人脸，resize 到 512 后直接推理（原逻辑）。
- whole：LMDB 为整图，走 FaceRestoreHelper 检测、对齐、还原与贴回（同 inference_codeformer.py）。
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
from basicsr.utils.misc import get_device, gpu_is_available
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper
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


def _set_realesrgan(args):
    """与 inference_codeformer.set_realesrgan 相同，但显式传入 args（避免依赖全局变量）。"""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available():
        no_half_gpu_list = ["1650", "1660"]
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=args.bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half,
    )

    if not gpu_is_available():
        import warnings

        warnings.warn(
            "Running on CPU now! RealESRGAN is slow on CPU. "
            "Consider omitting --bg_upsampler and --face_upsample.",
            category=RuntimeWarning,
        )
    return upsampler


def _run_codeformer_on_cropped(net, device, cropped_face: np.ndarray, w: float) -> np.ndarray:
    """单张对齐后人脸 BGR uint8 -> BGR uint8 还原结果。"""
    cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
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


def _restore_faces_on_helper(net, device, face_helper, w: float) -> None:
    """对 face_helper.cropped_faces 逐张推理并 add_restored_face。"""
    for cropped_face in face_helper.cropped_faces:
        restored_face = _run_codeformer_on_cropped(net, device, cropped_face, w)
        face_helper.add_restored_face(restored_face, cropped_face)


def main():
    parser = argparse.ArgumentParser(
        description="CodeFormer inference on LMDB (aligned crops or whole images with detect/align/paste)."
    )
    parser.add_argument("lmdb_path", type=str, help="LMDB 目录路径")
    parser.add_argument("-o", "--output", type=str, default="results_codeformer_lmdb", help="输出目录")
    parser.add_argument(
        "--input_mode",
        type=str,
        choices=["aligned", "whole"],
        default="aligned",
        help="aligned=已对齐人脸；whole=整图（检测、五点对齐、warp、贴回）",
    )
    parser.add_argument(
        "-w",
        "--fidelity_weight",
        type=float,
        default=0.5,
        help="Balance quality and fidelity (同 inference_codeformer -w). Default: 0.5",
    )
    parser.add_argument("-s", "--upscale", type=int, default=2, help="整图模式下的 upscale（同官方 -s）")
    parser.add_argument(
        "--only_center_face",
        action="store_true",
        help="整图模式：只还原位于图像中心的人脸",
    )
    parser.add_argument("--draw_box", action="store_true", help="整图模式：在结果上画人脸框")
    parser.add_argument(
        "--detection_model",
        type=str,
        default="retinaface_resnet50",
        help="整图模式人脸检测器，同 inference_codeformer",
    )
    parser.add_argument(
        "--bg_upsampler",
        type=str,
        default="None",
        help="整图模式背景超分：None | realesrgan",
    )
    parser.add_argument(
        "--face_upsample",
        action="store_true",
        help="整图模式：贴回前对人脸区域再超分",
    )
    parser.add_argument("--bg_tile", type=int, default=400, help="RealESRGAN tile 大小")
    parser.add_argument("--suffix", type=str, default=None, help="输出文件名后缀（同官方）")
    parser.add_argument("-n", "--max-images", type=int, default=10, help="最多推理条数（默认 10）")
    parser.add_argument("--ext", type=str, default="png", help="输出扩展名: png | jpg（仅 aligned 模式主输出）")
    parser.add_argument(
        "--save-restored-only",
        action="store_true",
        help="不保存对比图：aligned 下不写 cmp/；whole 下不写 cmp/ 内的拼图",
    )
    args = parser.parse_args()

    w = args.fidelity_weight
    whole = args.input_mode == "whole"

    if whole:
        os.makedirs(os.path.join(args.output, "cropped_faces"), exist_ok=True)
        os.makedirs(os.path.join(args.output, "restored_faces"), exist_ok=True)
        os.makedirs(os.path.join(args.output, "final_results"), exist_ok=True)
        if not args.save_restored_only:
            os.makedirs(os.path.join(args.output, "cmp"), exist_ok=True)
    else:
        os.makedirs(os.path.join(args.output, "restored_imgs"), exist_ok=True)
        if not args.save_restored_only:
            os.makedirs(os.path.join(args.output, "cmp"), exist_ok=True)

    if args.bg_upsampler == "realesrgan":
        bg_upsampler = _set_realesrgan(args)
    else:
        bg_upsampler = None

    if args.face_upsample:
        face_upsampler = bg_upsampler if bg_upsampler is not None else _set_realesrgan(args)
    else:
        face_upsampler = None

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

    if whole:
        print(f"Face detection model: {args.detection_model}")
        print(f"Background upsampling: {bg_upsampler is not None}, Face upsampling: {args.face_upsample}")

    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=args.detection_model,
        save_ext="png",
        use_parse=True,
        device=device,
    )

    engine = LMDBEngine(args.lmdb_path, write=False)
    try:
        all_keys = engine.keys()
        keys = all_keys[: args.max_images]
        for key in keys:
            print(f"Processing {key} ...")
            img_t = engine[key]
            img = _tensor_chw_rgb_to_bgr_uint8(img_t)
            if is_gray(img, threshold=10):
                print("\tGrayscale-like input: True")

            safe_key = key.replace("/", "_")
            basename = safe_key
            if "." in basename:
                basename = os.path.splitext(basename)[0]

            face_helper.clean_all()

            if whole:
                face_helper.read_image(img)
                num_det = face_helper.get_face_landmarks_5(
                    only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5
                )
                print(f"\tdetect {num_det} faces")
                if num_det == 0:
                    print(f"\tSkip {key}: no face detected.")
                    continue
                face_helper.align_warp_face()
            else:
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_helper.is_gray = is_gray(img, threshold=10)
                if face_helper.is_gray:
                    print("\tGrayscale input: True")
                face_helper.cropped_faces = [img]

            _restore_faces_on_helper(net, device, face_helper, w)

            if whole:
                if bg_upsampler is not None:
                    bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
                else:
                    bg_img = None
                face_helper.get_inverse_affine(None)
                if args.face_upsample and face_upsampler is not None:
                    restored_img = face_helper.paste_faces_to_input_image(
                        upsample_img=bg_img, draw_box=args.draw_box, face_upsampler=face_upsampler
                    )
                else:
                    restored_img = face_helper.paste_faces_to_input_image(
                        upsample_img=bg_img, draw_box=args.draw_box
                    )

                for idx, (cropped_face, restored_face) in enumerate(
                    zip(face_helper.cropped_faces, face_helper.restored_faces)
                ):
                    save_crop_path = os.path.join(
                        args.output, "cropped_faces", f"{basename}_{idx:02d}.png"
                    )
                    imwrite(cropped_face, save_crop_path)
                    save_face_name = f"{basename}_{idx:02d}.png"
                    if args.suffix is not None:
                        save_face_name = f"{save_face_name[:-4]}_{args.suffix}.png"
                    imwrite(
                        restored_face,
                        os.path.join(args.output, "restored_faces", save_face_name),
                    )
                    if not args.save_restored_only:
                        cmp_face = np.concatenate((cropped_face, restored_face), axis=1)
                        imwrite(cmp_face, os.path.join(args.output, "cmp", save_face_name))

                if restored_img is not None:
                    final_base = basename if args.suffix is None else f"{basename}_{args.suffix}"
                    imwrite(
                        restored_img,
                        os.path.join(args.output, "final_results", f"{final_base}.png"),
                    )
                    if not args.save_restored_only:
                        # 整图对比：缩放到相同高度后横向拼接
                        target_h = min(img.shape[0], restored_img.shape[0], 720)
                        scale_a = target_h / img.shape[0]
                        scale_b = target_h / restored_img.shape[0]
                        wa = int(img.shape[1] * scale_a)
                        wb = int(restored_img.shape[1] * scale_b)
                        left = cv2.resize(img, (wa, target_h), interpolation=cv2.INTER_LINEAR)
                        right = cv2.resize(restored_img, (wb, target_h), interpolation=cv2.INTER_LINEAR)
                        imwrite(
                            np.concatenate((left, right), axis=1),
                            os.path.join(args.output, "cmp", f"{final_base}__full.png"),
                        )
            else:
                restored_face = face_helper.restored_faces[0]
                out_name = f"{safe_key}.{args.ext}"
                if not args.save_restored_only:
                    inp512 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                    cmp_img = np.concatenate((inp512, restored_face), axis=1)
                    imwrite(cmp_img, os.path.join(args.output, "cmp", out_name))
                imwrite(restored_face, os.path.join(args.output, "restored_imgs", out_name))
    finally:
        engine.close()

    if whole:
        print(
            f"Done (whole). cropped_faces / restored_faces / final_results under [{args.output}]"
        )
    elif args.save_restored_only:
        print(f"Done. Restored images under [{os.path.join(args.output, 'restored_imgs')}]")
    else:
        print(
            f"Done. Comparison (输入512|还原) under [{os.path.join(args.output, 'cmp')}], "
            f"restored under [{os.path.join(args.output, 'restored_imgs')}]"
        )


if __name__ == "__main__":
    main()
