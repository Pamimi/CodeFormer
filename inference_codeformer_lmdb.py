#!/usr/bin/env python3
"""CodeFormer 对 LMDB 中图像推理；流程与 inference_codeformer.py 一致，仅输入改为 LMDB。"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

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

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

pretrain_model_url = {
    "restoration": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
}


def _sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _lmdb_tensor_to_bgr(img_t: torch.Tensor) -> np.ndarray:
    if img_t.dim() != 3:
        raise ValueError(f"expect CHW image tensor, got {tuple(img_t.shape)}")
    x = img_t.detach().cpu()
    if x.dtype != torch.uint8:
        x = x.clamp(0, 255).to(torch.uint8)
    rgb = x.permute(1, 2, 0).numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    device = get_device()
    parser = argparse.ArgumentParser()
    parser.add_argument("lmdb_path", type=str, help="LMDB 目录")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="输出目录，默认 results/<lmdb名>_<w>")
    parser.add_argument("-w", "--fidelity_weight", type=float, default=0.5)
    parser.add_argument("-s", "--upscale", type=int, default=2)
    parser.add_argument("-n", "--max_images", type=int, default=10, help="最多处理条数")
    parser.add_argument("--has_aligned", action="store_true", help="LMDB 为已对齐人脸 crop")
    parser.add_argument("--only_center_face", action="store_true")
    parser.add_argument("--draw_box", action="store_true")
    parser.add_argument("--detection_model", type=str, default="retinaface_resnet50")
    parser.add_argument("--bg_upsampler", type=str, default="None", help="None | realesrgan")
    parser.add_argument("--face_upsample", action="store_true")
    parser.add_argument("--bg_tile", type=int, default=400)
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()

    w = args.fidelity_weight
    lmdb_name = os.path.basename(os.path.normpath(args.lmdb_path))
    result_root = args.output_path if args.output_path else f"results/{lmdb_name}_{w}"

    def set_realesrgan():
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
                "Running on CPU now! RealESRGAN is slow on CPU.",
                category=RuntimeWarning,
            )
        return upsampler

    if args.bg_upsampler == "realesrgan":
        bg_upsampler = set_realesrgan()
    else:
        bg_upsampler = None
    if args.face_upsample:
        face_upsampler = bg_upsampler if bg_upsampler is not None else set_realesrgan()
    else:
        face_upsampler = None

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

    if not args.has_aligned:
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

    os.makedirs(os.path.join(result_root, "restored_faces"), exist_ok=True)
    if not args.has_aligned:
        os.makedirs(os.path.join(result_root, "cropped_faces"), exist_ok=True)
        os.makedirs(os.path.join(result_root, "final_results"), exist_ok=True)

    lmdb_env = lmdb.open(
        args.lmdb_path, readonly=True, lock=False, readahead=False, meminit=True
    )
    txn = lmdb_env.begin(write=False)
    all_keys = [k.decode() for k in txn.cursor().iternext(values=False)]
    input_keys = all_keys[: args.max_images]
    test_img_num = len(input_keys)
    if test_img_num == 0:
        lmdb_env.close()
        raise FileNotFoundError("LMDB 中无 key")

    stage_labels = (
        ("decode", "LMDB取数+解码转BGR"),
        ("preprocess", "人脸分支预处理"),
        ("landmarks", "人脸检测与五点(get_landmarks5)"),
        ("align", "对齐裁剪(align_warp_face)"),
        ("codeformer", "CodeFormer(全部人脸)"),
        ("bg_sr", "RealESRGAN背景"),
        ("paste", "逆仿射+贴回(paste_faces)"),
        ("save", "写盘(imwrite)"),
    )
    timing_rows = []

    try:
        for i, key in enumerate(input_keys):
            face_helper.clean_all()
            basename = key.replace("/", "_")
            if "." in basename:
                basename = os.path.splitext(basename)[0]
            print(f"[{i + 1}/{test_img_num}] {key}")
            st = {k: 0.0 for k, _ in stage_labels}

            t0 = time.perf_counter()
            payload = txn.get(key.encode())
            if payload is None:
                print(f"\tSkip: key missing")
                continue
            try:
                buf = torch.tensor(np.frombuffer(payload, dtype=np.uint8))
                img_t = torchvision.io.decode_image(buf, mode=torchvision.io.ImageReadMode.RGB)
            except Exception:
                img_t = torch.load(io.BytesIO(payload), weights_only=True)
            img = _lmdb_tensor_to_bgr(img_t)
            _sync_device()
            st["decode"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            if args.has_aligned:
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_helper.is_gray = is_gray(img, threshold=10)
                if face_helper.is_gray:
                    print("\tGrayscale input: True")
                face_helper.cropped_faces = [img]
            else:
                face_helper.read_image(img)
            st["preprocess"] = (time.perf_counter() - t0) * 1000.0

            if not args.has_aligned:
                t0 = time.perf_counter()
                num_det = face_helper.get_face_landmarks_5(
                    only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5
                )
                _sync_device()
                st["landmarks"] = (time.perf_counter() - t0) * 1000.0
                print(f"\tdetect {num_det} faces")
                if num_det == 0:
                    print(
                        f"\t阶段(ms): LMDB取数+解码转BGR={st['decode']:.1f} "
                        f"预处理={st['preprocess']:.1f} 人脸检测={st['landmarks']:.1f}"
                    )
                    continue
                t0 = time.perf_counter()
                face_helper.align_warp_face()
                st["align"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            for cropped_face in face_helper.cropped_faces:
                cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
                try:
                    with torch.no_grad():
                        output = net(cropped_face_t, w=w, adain=True)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                except Exception as error:
                    print(f"\tFailed inference for CodeFormer: {error}")
                    restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
                restored_face = restored_face.astype("uint8")
                face_helper.add_restored_face(restored_face, cropped_face)
            _sync_device()
            st["codeformer"] = (time.perf_counter() - t0) * 1000.0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            restored_img = None
            if not args.has_aligned:
                t0 = time.perf_counter()
                if bg_upsampler is not None:
                    bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
                else:
                    bg_img = None
                _sync_device()
                st["bg_sr"] = (time.perf_counter() - t0) * 1000.0
                t0 = time.perf_counter()
                face_helper.get_inverse_affine(None)
                if args.face_upsample and face_upsampler is not None:
                    restored_img = face_helper.paste_faces_to_input_image(
                        upsample_img=bg_img, draw_box=args.draw_box, face_upsampler=face_upsampler
                    )
                else:
                    restored_img = face_helper.paste_faces_to_input_image(
                        upsample_img=bg_img, draw_box=args.draw_box
                    )
                _sync_device()
                st["paste"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            for idx, (cropped_face, restored_face) in enumerate(
                zip(face_helper.cropped_faces, face_helper.restored_faces)
            ):
                if not args.has_aligned:
                    imwrite(
                        cropped_face,
                        os.path.join(result_root, "cropped_faces", f"{basename}_{idx:02d}.png"),
                    )
                if args.has_aligned:
                    save_face_name = f"{basename}.png"
                else:
                    save_face_name = f"{basename}_{idx:02d}.png"
                if args.suffix is not None:
                    save_face_name = f"{save_face_name[:-4]}_{args.suffix}.png"
                imwrite(
                    restored_face,
                    os.path.join(result_root, "restored_faces", save_face_name),
                )

            if not args.has_aligned and restored_img is not None:
                fb = basename if args.suffix is None else f"{basename}_{args.suffix}"
                imwrite(
                    restored_img,
                    os.path.join(result_root, "final_results", f"{fb}.png"),
                )
            st["save"] = (time.perf_counter() - t0) * 1000.0

            total_ms = sum(st.values())
            print(
                "\t阶段(ms): "
                + " ".join(f"{lab}={st[k]:.1f}" for k, lab in stage_labels)
                + f" | 合计={total_ms:.1f}"
            )
            timing_rows.append(st)

    finally:
        lmdb_env.close()

    print(f"\nAll results are saved in {result_root}")

    if timing_rows:
        n = len(timing_rows)
        print(f"\n--- 阶段耗时汇总 (成功 {n} 条, 算术平均 ms) ---")
        for k, lab in stage_labels:
            avg = sum(r[k] for r in timing_rows) / n
            if avg > 0.05 or k in ("decode", "codeformer", "save"):
                print(f"  {lab}: {avg:.1f}")
