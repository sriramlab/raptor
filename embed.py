
import os, sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
import shutil
import zipfile
import re
from time import time
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

LATENT_SIZE_LOOKUP = dict(
    SAM=256,
    MedSAM=256,
    CLIP=1024,
    DINO=1024,
    DINOv2=1024,
    DINOv3=4096,
    LlavaMed=1024,
)

def get_prefix_token_count(image_encoder):
    config = getattr(image_encoder, 'config', None)
    if config is None:
        return 0

    # ViT-style Hugging Face vision backbones return CLS/register prefix tokens
    # before spatial patch tokens. Raptor should use patch tokens for dense maps.
    has_cls = any(hasattr(config, name) for name in ['num_register_tokens', 'patch_size', 'image_size'])
    if not has_cls:
        return 0
    return 1 + int(getattr(config, 'num_register_tokens', 0) or 0)

MODEL_URL_LOOKUP = dict(
    DINO='facebook/dinov2-large',
    DINOv2='facebook/dinov2-large',
    DINOv3='facebook/dinov3-vit7b16-pretrain-lvd1689m',
    CLIP='openai/clip-vit-large-patch14',
)

def get_image_encoder(encoder_name, device):

    preprocessor = lambda imgs: imgs
    if encoder_name in ['MedSAM', 'SAM']:
        sys.path.append('../etc/MedSAM')
        from segment_anything import sam_model_registry
        sam_checkpoints = dict(
            MedSAM='../etc/MedSAM/work_dir/MedSAM/medsam_vit_b.pth',
            SAM='../etc/MedSAM/work_dir/MedSAM/sam_vit_b_01ec64.pth',
        )
        image_encoder = sam_model_registry['vit_b'](checkpoint=sam_checkpoints[encoder_name])

    elif encoder_name in MODEL_URL_LOOKUP:
        model_url = MODEL_URL_LOOKUP[encoder_name]

        processor = AutoImageProcessor.from_pretrained(model_url, use_fast=True)
        model_kwargs = {}
        if encoder_name == 'DINOv3' and str(device).startswith('cuda'):
            model_kwargs['torch_dtype'] = torch.float16
        image_encoder = AutoModel.from_pretrained(model_url, **model_kwargs).to(device)

        if encoder_name == 'CLIP':
            image_encoder = image_encoder.vision_model

        preprocessor = lambda imgs: processor(images=[Image.fromarray((np.clip(i, 0, 1)*255).astype(np.uint8)) for i in imgs], return_tensors="pt").to(device)

    elif encoder_name == 'LlavaMed':
        sys.path.append('../etc/LLaVA-Med')
        from llava.model.builder import load_pretrained_model
        tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path='/u/scratch/u/ulzee/hug/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/f2f72301dc934e74948b5802c87dbc83d100e6bd/',
                model_base=None,
                model_name='llava-med-v1.5-mistral-7b'
        )
        image_encoder = model.get_vision_tower()

    else:
        raise ValueError(f'Unknown model: {encoder_name}')

    return preprocessor, image_encoder

def normalize_volume(vol, mode):
    vol = np.asarray(vol, dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == 'ukbb_mri':
        hclip = 1024 + 256
        vol[vol < 0] = 0
        vol[vol > hclip] = hclip
        vol /= hclip
        return vol

    if mode == 'medmnist':
        vol /= 256
        return np.clip(vol, 0, 1)

    if mode == 'oct':
        lo, hi = np.percentile(vol, [0.5, 99.5])
        if not np.isfinite(hi) or hi <= lo:
            lo, hi = float(vol.min()), float(vol.max())
        if hi <= lo:
            return np.zeros_like(vol, dtype=np.float32)
        vol = np.clip(vol, lo, hi)
        vol = (vol - lo) / (hi - lo)
        return vol.astype(np.float32)

    raise ValueError(f'Unknown normalization mode: {mode}')

def load_oct_npz(path):
    with np.load(path, allow_pickle=True) as zf:
        if 'oct_volume_normalized' in zf.files:
            vol = zf['oct_volume_normalized'].astype(np.float32)
            if vol.ndim != 3:
                raise ValueError(f'{path} volume must be 3D, got shape {vol.shape}')
            return np.clip(vol, 0, 1)
        if 'oct_volume' in zf.files:
            vol = zf['oct_volume']
        elif len(zf.files) == 1:
            vol = zf[zf.files[0]]
        else:
            raise ValueError(f'{path} has no oct_volume key; keys={zf.files}')

    if vol.ndim != 3:
        raise ValueError(f'{path} volume must be 3D, got shape {vol.shape}')
    return normalize_volume(vol, 'oct')

def natural_key(path):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', os.path.basename(path))]

def load_oct_image_folder(path):
    exts = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
    files = [
        os.path.join(path, fname)
        for fname in os.listdir(path)
        if fname.lower().endswith(exts)
    ]
    files = sorted(files, key=natural_key)
    if not files:
        raise ValueError(f'{path} has no OCT image slices')

    slices = []
    expected_shape = None
    for fname in files:
        arr = np.asarray(Image.open(fname).convert('L'), dtype=np.float32)
        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise ValueError(f'{path} has mixed slice shapes: {expected_shape} and {arr.shape} at {fname}')
        slices.append(arr)

    vol = np.stack(slices, axis=0)
    return normalize_volume(vol, 'oct')

def load_oct_image_zip(path):
    exts = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
    with zipfile.ZipFile(path, 'r') as zf:
        files = [
            name for name in zf.namelist()
            if not name.endswith('/') and name.lower().endswith(exts)
        ]
        files = sorted(files, key=natural_key)
        if not files:
            raise ValueError(f'{path} has no OCT image slices')

        slices = []
        expected_shape = None
        for fname in files:
            with zf.open(fname) as fl:
                arr = np.asarray(Image.open(fl).convert('L'), dtype=np.float32)
            if expected_shape is None:
                expected_shape = arr.shape
            elif arr.shape != expected_shape:
                raise ValueError(f'{path} has mixed slice shapes: {expected_shape} and {arr.shape} at {fname}')
            slices.append(arr)

    vol = np.stack(slices, axis=0)
    return normalize_volume(vol, 'oct')

def load_oct_path(path):
    if os.path.isdir(path):
        return load_oct_image_folder(path)
    if os.path.isfile(path) and path.lower().endswith('.npz'):
        return load_oct_npz(path)
    if os.path.isfile(path) and path.lower().endswith('.zip'):
        return load_oct_image_zip(path)
    raise ValueError(f'{path} is not a supported OCT path')

def equidistant_subsample_volume(vol, factor):
    if factor is None or factor <= 1:
        return vol
    if vol.ndim != 3:
        raise ValueError(f'Expected 3D volume for subsampling, got shape {vol.shape}')

    indices = []
    for dim in vol.shape:
        n = max(1, int(np.ceil(dim / factor)))
        idx = np.linspace(0, dim - 1, n, dtype=np.int64)
        indices.append(np.unique(idx))
    return vol[np.ix_(indices[0], indices[1], indices[2])]

def save_name_for_pid(pid):
    name = os.path.splitext(pid)[0]
    if os.path.isabs(pid):
        name = name.lstrip(os.sep)
    else:
        name = os.path.basename(name)
    return re.sub(r'[^A-Za-z0-9._-]+', '__', name)

def output_proj_name(projname, planes):
    if len(planes) < 3:
        return f'p{"".join(planes)}_{projname}'
    return projname

def crop_pad_matrix(mat, size=224):
    if all([dim == size for dim in mat.shape]):
        return mat

    mat = torch.nn.functional.interpolate(
        torch.from_numpy(mat[np.newaxis, np.newaxis, :, :].astype(np.float32)), size=(size, size),
        mode='bicubic',
        align_corners=False,
    ).squeeze().numpy()
    return mat

def resize_pad_matrix(mat, size=224, pad_value=0.0, scale=None):
    mat = mat.astype(np.float32, copy=False)
    h, w = mat.shape
    if h <= 0 or w <= 0:
        raise ValueError(f'Cannot resize empty 2D slice with shape {mat.shape}')

    if scale is None:
        scale = min(size / h, size / w)

    new_h = max(1, min(size, int(round(h * scale))))
    new_w = max(1, min(size, int(round(w * scale))))
    mat = torch.nn.functional.interpolate(
        torch.from_numpy(mat[np.newaxis, np.newaxis, :, :]),
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=False,
    ).squeeze().numpy()

    out = np.full((size, size), pad_value, dtype=np.float32)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    out[top:top + new_h, left:left + new_w] = mat
    return out

def preprocess_slice_matrix(mat, mode, size=224, global_scale=None):
    if mode == 'stretch':
        return crop_pad_matrix(mat, size=size)
    if mode == 'pad':
        return resize_pad_matrix(mat, size=size)
    if mode == 'global_pad':
        if global_scale is None:
            raise ValueError('global_pad requires global_scale')
        return resize_pad_matrix(mat, size=size, scale=global_scale)
    raise ValueError(f'Unknown resize mode: {mode}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--folder', type=str, default=None, help='Folder containing scans in .nii.gz or .zip (UKBB style)')
    parser.add_argument('--npz', type=str, default=None, help='Npz blob containing {train,val,test}_images fields')
    parser.add_argument('--oct_npz_folder', type=str, default=None, help='Folder containing one OCT .npz volume per manifest entry')
    parser.add_argument('--oct_image_folder', type=str, default=None, help='Folder containing one OCT image-slice folder per manifest entry')
    parser.add_argument('--oct_path_list', default=False, action='store_true', help='Manifest entries are absolute OCT .npz files, image-slice folders, or zip files')
    parser.add_argument('--extract_file', type=str, default='T1/T1_brain.nii.gz', help='The nii.gz to read if given zip files')
    parser.add_argument('--encoder', type=str, default='DINO', help='Encoder type')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--manifest', type=str, required=True, help='Manifest file path')
    parser.add_argument('--start', type=int, required=True, help='Start index')
    parser.add_argument('--many', type=int, required=True, help='Number of files to process')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_torch_threads', type=int, default=4, help='Number of CPU threads used by torch')
    parser.add_argument('--saveto', type=str, required=True, help='Save directory')
    parser.add_argument('--k', type=str, default=None)
    parser.add_argument('--planes', default='A,C,S')
    parser.add_argument('--avgpool', default=False, action='store_true')
    parser.add_argument('--skip_existing', default=False, action='store_true')
    parser.add_argument('--continue_on_error', default=False, action='store_true')
    parser.add_argument('--errors_log', type=str, default=None)
    parser.add_argument('--keep_prefix_tokens', default=False, action='store_true', help='Keep CLS/register tokens in transformer outputs')
    parser.add_argument('--subsample_factor', type=int, default=1, help='Equidistantly subsample each volume axis by this factor before slicing')
    parser.add_argument('--resize_mode', choices=['stretch', 'pad', 'global_pad'], default='stretch',
                        help='2D slice resize policy: old stretch-to-square, per-slice aspect pad, or one global volume scale plus pad')

    args = parser.parse_args()

    torch.set_num_threads(args.num_torch_threads)

    args.planes = args.planes.split(',') if ',' in args.planes else list(args.planes)
    for p in args.planes:
        assert p in 'ACS'
    if args.k is not None:
        args.k = args.k.split(',')
        print(f'Loading {len(args.k)} projections...')
    if args.avgpool:
        assert args.k is None # there is no need for Ks
    else:
        assert args.k is not None

    if sum([args.folder is not None, args.npz is not None, args.oct_npz_folder is not None, args.oct_image_folder is not None, args.oct_path_list]) != 1:
        raise ValueError('Exactly one of --folder, --npz, --oct_npz_folder, --oct_image_folder, or --oct_path_list is required')

    with open(args.manifest) as fl:
        fls = [ln.strip() for ln in fl if ln.strip()]
    fbatch = fls[args.start:args.start+args.many]

    # If MedMNIST format npz, unpack relevant images first
    npzcache = dict()
    npzblob = None
    if args.npz:
        npzblob = np.load(args.npz)
        for split in ['train', 'test', 'val']:
            npzcache[split] = dict()
            fs = [f for f in fbatch if split in f]
            fixs = [int(f.split('_')[1]) for f in fs]
            if len(fixs) == 0: continue
            fimin, fimax = min(fixs), max(fixs)
            batch = npzblob[f'{split}_images'][fimin:fimax+1]
            for fi in range(fimin, fimax+1):
                npzcache[split][fi] = batch[fi-fimin]

    preprocessor, image_encoder = get_image_encoder(args.encoder, args.device)
    image_encoder = image_encoder.to(args.device).eval()
    prefix_tokens = 0 if args.keep_prefix_tokens else get_prefix_token_count(image_encoder)
    print(f'Dropping {prefix_tokens} prefix tokens per slice')

    projfiles = []
    if args.k is not None:
        for fname in args.k:
            projfiles += [(fname.split('/')[-1].split('.')[0], np.load(fname))]
    else:
        projfiles = [('proj_identity', None)]

    pbar = tqdm(fbatch)
    for pid in pbar:
        pid_save_name = save_name_for_pid(pid)
        if args.skip_existing:
            expected = [
                f'{args.saveto}/{output_proj_name(projname, args.planes)}/{pid_save_name}.npy'
                for projname, _ in projfiles
            ]
            if all(os.path.exists(fname) for fname in expected):
                pbar.set_postfix(dict(pid=pid, skipped=True))
                continue

        try:
            if args.folder is not None and '.zip' in pid:
                zipname = f'{args.folder}/{pid}'


                temp_folder = f'temp/{pid}'
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)
                os.makedirs(temp_folder, exist_ok=True)
                with zipfile.ZipFile(zipname, 'r') as zip_ref:
                    if args.extract_file in zip_ref.namelist():
                        zip_ref.extract(args.extract_file, path=temp_folder)
                        file_path = f'{temp_folder}/{args.extract_file}'
                    else:
                        continue

                import nibabel as nib
                vol = nib.load(file_path).get_fdata()
                shutil.rmtree(temp_folder)

                vol = normalize_volume(vol, 'ukbb_mri')
            elif args.npz:
                vol = npzcache[pid.split('_')[0]][int(pid.split('_')[1])]
                vol = normalize_volume(vol, 'medmnist')
            elif args.oct_npz_folder:
                file_path = pid if os.path.isabs(pid) else f'{args.oct_npz_folder}/{pid}'
                vol = load_oct_npz(file_path)
            elif args.oct_image_folder:
                file_path = pid if os.path.isabs(pid) else f'{args.oct_image_folder}/{pid}'
                vol = load_oct_image_folder(file_path)
            elif args.oct_path_list:
                vol = load_oct_path(pid)
            else:
                raise ValueError(f'Unsupported manifest entry for selected input mode: {pid}')
        except Exception as exc:
            if not args.continue_on_error:
                raise
            msg = f'{pid}\t{type(exc).__name__}\t{exc}'
            pbar.write(msg)
            if args.errors_log:
                os.makedirs(os.path.dirname(args.errors_log), exist_ok=True)
                with open(args.errors_log, 'a') as err_fl:
                    err_fl.write(msg + '\n')
            continue

        try:
            vol = equidistant_subsample_volume(vol, args.subsample_factor)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            msg = f'{pid}\t{type(exc).__name__}\tsubsample failed: {exc}'
            pbar.write(msg)
            if args.errors_log:
                os.makedirs(os.path.dirname(args.errors_log), exist_ok=True)
                with open(args.errors_log, 'a') as err_fl:
                    err_fl.write(msg + '\n')
            continue

        # collect slices (in axes order)
        slices_byxyz = []

        ntot = []
        if 'A' in args.planes:
            slices = []
            for i in range(0, vol.shape[0]-0):
                if vol[i].shape[0] == 1 or vol[i].shape[1] == 1: continue
                slices += [vol[i]]
            nx = len(slices)
            ntot += [nx]
            slices_byxyz += slices

        if 'C' in args.planes:
            slices = []
            for i in range(0, vol.shape[1]-0):
                if vol[:, i].shape[0] == 1 or vol[:, i].shape[1] == 1: continue
                slices += [vol[:, i]]
            ny = len(slices)
            ntot += [ny]
            slices_byxyz += slices

        if 'S' in args.planes:
            slices = []
            for i in range(0, vol.shape[2]-0):
                if vol[:, :, i].shape[0] == 1 or vol[:, :, i].shape[1] == 1: continue
                slices += [vol[:, :, i]]
            nz = len(slices)
            ntot += [nz]
            slices_byxyz += slices

        t0 = time()
        global_scale = None
        if args.resize_mode == 'global_pad':
            global_scale = min(224.0 / dim for dim in vol.shape)
        imgs = [
            preprocess_slice_matrix(img, args.resize_mode, global_scale=global_scale)
            for img in slices_byxyz
        ]
        t_crop = time() - t0
        embs = []
        t0 = time()
        for i in range(0, len(imgs), args.batch_size):
            inputs = preprocessor(imgs[i:i+args.batch_size])

            with torch.no_grad():
                outputs = image_encoder(**inputs)
                last_hidden_states = outputs.last_hidden_state
                if prefix_tokens:
                    last_hidden_states = last_hidden_states[:, prefix_tokens:, :]
                embs += [e.T for e in last_hidden_states.detach().cpu().numpy()]

        savetimes = []
        for (projname, projmat) in projfiles:
            t0 = time()
            if projmat is not None:
                projname = output_proj_name(projname, args.planes)

                # projmat: D x K         (D: ViT dim, K: projections)
                # embs: S x D x 16 x 16  (S: slices)
                assert len(projmat) == len(embs[0])

            if not os.path.exists(f'{args.saveto}/{projname}'):
                os.makedirs(f'{args.saveto}/{projname}')

            # proj_embs: slices (16 + 16 + 16) x K x 256
            # proj_embs_sum: slices 3 x K x 16 x 16
            assert np.sum(ntot) == len(embs)

            if args.avgpool:
                proj_embs = proj_embs.reshape(len(proj_embs), LATENT_SIZE_LOOKUP[args.encoder], -1)
                proj_embs_sum_flat = proj_embs.mean((0, -1))
            else:
                plane_breaks = []
                agg = 0
                for s_count in ntot:
                    plane_breaks += [s_count + agg]
                    agg += s_count

                byside = [s for s in np.split(embs, plane_breaks, axis=0) if len(s)]
                byside = [s.mean(0) for s in byside]
                byside = [projmat.T @ s.reshape(projmat.shape[0], -1) for s in byside]
                assert len(byside) == len(args.planes)
                proj_embs_sum = np.concatenate(byside)

                # proj_embs_sum: slices 3K x 256 ~ 7680 for K=10
                proj_embs_sum_flat = proj_embs_sum.reshape(-1)

            t0 = time()
            np.save(f'{args.saveto}/{projname}/{pid_save_name}.npy', proj_embs_sum_flat.astype(np.float32))
            savetimes += [time() - t0]

        pbar.set_postfix(dict(
            pid=pid, sh=vol.shape[0], ns=len(slices_byxyz), d=proj_embs_sum_flat.shape,
        ))
