import itertools
import os
import torch
from tqdm import tqdm
from monai.networks.nets.autoencoderkl import AutoencoderKL
from model import ASLDM
from utils import visualize_2d_image
from utils import read_dir
import PIL
import monai.transforms as mt
from monai.data import DataLoader, Dataset
from monai.networks.schedulers.pndm import PNDMScheduler
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from monai.utils import first, set_determinism
import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error as mae
import ants
import numpy as np
def np_to_ants(img_np):
    np_img = np.array(img_np).astype(np.float32)
    ants_img = ants.from_numpy(np_img)
    return ants_img


def ants_to_tensor(ants_img):
    np_img = ants_img.numpy()
    tensor_img = torch.tensor(np_img)
    return tensor_img

if __name__ == "__main__":
    seed = 42
    num_workers = 0
    # 设置随机种子
    set_determinism(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_variations = list(
        itertools.product(
            *[(c, "1") if c == "0" else (c, "0") for c in ["0", "0", "0", "0", "0"]]
        )
    )
    # 过滤掉全0和全1的变体（虽然在这个特定情况下不会出现全0或全1）
    valid_variations = [var for var in all_variations if "1" in var and "0" in var]
    for missing_code_idx in range(len(valid_variations)):
        missing_code = valid_variations[missing_code_idx]
        missing_code_str = "".join(missing_code)
        print(f"Processing variation {missing_code_idx}: {missing_code_str}")
        modal_map = {
            "10000": "t1n",
            "01000": "t2w",
            "00100": "t1c",
            "00010": "dwi",
            "00001": "adc",
        }
        input_modal_code = "11111"
        output_modal_code = "11111"
        input_modals = [
            modal_map[code]
            for code, bit in zip(modal_map.keys(), input_modal_code)
            if bit == "1"
        ]
        output_modals = [
            modal_map[code]
            for code, bit in zip(modal_map.keys(), output_modal_code)
            if bit == "1"
        ]
        aux_modalities = ["seg", "sketch"]
        all_modalities = input_modals + aux_modalities
        # data_dir = "/home/langtaoz/projects/data/OTTS_2D_MIMO_png"
        data_dir = "data"
        test_datalist = []
        # 获取所有模态的图像文件名
        test_modal_paths = [f"{data_dir}/test/{modal}" for modal in all_modalities]
        # 读取每个模态的图像文件路径
        test_modal_files = [
            read_dir(path, lambda x: x.endswith(".png"), recursive=True)
            for path in test_modal_paths
        ]

        # 构建数据列表
        for file_idx in range(len(test_modal_files[0])):
            temp_obj = {}
            for modal_idx, modal in enumerate(all_modalities):
                temp_obj[modal] = test_modal_files[modal_idx][file_idx]
            test_datalist.append(temp_obj)
        test_transforms = mt.Compose(
            [
                mt.LoadImaged(keys=all_modalities),
                mt.EnsureChannelFirstd(keys=all_modalities),
                mt.ScaleIntensityRanged(
                    keys=all_modalities,
                    a_min=0.0,
                    a_max=255.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )
        test_dataset = Dataset(data=test_datalist, transform=test_transforms)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
        )
        autoencoder = AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            latent_channels=1,
            channels=[64, 128, 256],
            num_res_blocks=2,
            norm_num_groups=32,
            norm_eps=1e-06,
            attention_levels=[False, False, False],
            with_encoder_nonlocal_attn=True,
            with_decoder_nonlocal_attn=True,
            include_fc=False,
        )
        autoencoder.load_state_dict(
            torch.load(
                "checkpoints/autoencoder.pt",
                weights_only=True,
            )
        )
        autoencoder.eval()
        autoencoder.to(device)

        unet_config = {
            "spatial_dims": 2,
            "in_channels": len(input_modals),
            "out_channels": len(input_modals),
            "num_res_blocks": 4,
            "channels": (128, 256, 512, 1024),
            "attention_levels": (False, True, True, True),
            "num_head_channels": (0, 32, 64, 128),
            "resblock_updown": True,
            "with_conditioning": True,
            "cross_attention_dim": 768,
        }
        unet = ASLDM(unet_config, num_modalities=len(input_modals))
        unet_pt = torch.load(
            f"checkpoints/diffusion_unet.pt",
            weights_only=True,
        )
        unet.load_state_dict(unet_pt)
        unet.eval()
        unet.to(device)
        
        scale_factor = 0.9708808064460754
        noise_scheduler = PNDMScheduler(
            num_train_timesteps=1000,
            schedule="scaled_linear_beta",
            beta_start=0.0015,
            beta_end=0.0195,
            skip_prk_steps=True,
        )
        noise_scheduler.set_timesteps(num_inference_steps=50)

        results_dict = {}
        target_batch_indices = range(0, len(test_dataloader))
        # target_batch_indices = range(420, 440)

        for idx, batch in tqdm(
            enumerate(test_dataloader), total=len(test_dataloader), desc="Testing"
        ):
            if idx not in target_batch_indices:
                continue
            with torch.no_grad():
                with torch.autocast("cuda", enabled=True):
                    source_images_list = [batch[modal].to(device) for modal in input_modals]
                    source_images = torch.cat(source_images_list, dim=1)
                    target_images_list = [
                        batch[modal].to(device) for modal in output_modals
                    ]
                    target_images = torch.cat(target_images_list, dim=1)
                    seg = batch["seg"].to(device)
                    sketch = batch["sketch"].to(device)

                    # 转成tensor形式，0/1整型
                    missing_code_tensor = (
                        torch.tensor(
                            [int(bit) for bit in missing_code],
                            dtype=torch.long,
                            device=device,
                        )
                        .unsqueeze(0)
                        .repeat(len(batch["seg"]), 1)
                    )  # 扩展成 [B, N]

                    source_latents_list = []
                    mask_list = []
                    for i, modal in enumerate(input_modals):
                        is_present = missing_code[i] == "1"
                        clean_latent = (
                            autoencoder.encode_stage_2_inputs(batch[modal].to(device))
                            * scale_factor
                        )
                        if is_present:
                            source_latents_list.append(clean_latent)
                            mask_list.append(torch.zeros_like(clean_latent))
                        else:
                            noisy_latent = torch.randn_like(clean_latent)
                            source_latents_list.append(noisy_latent)
                            mask_list.append(torch.ones_like(clean_latent))
                    source_latents = torch.cat(source_latents_list, dim=1)
                    mask_target = torch.cat(mask_list, dim=1)

                    # 初始化随机噪声（单个样本）
                    current_latents = source_latents.clone()

                    # 准备进度条（仅单个样本）
                    progress_bar = tqdm(noise_scheduler.timesteps, desc="Denoising")
                    denoising_chain = []
                    print(missing_code)
                    for t in progress_bar:
                        # 在每个时间步中，将未缺失模态的原始信息与当前的 current_latents 结合
                        combined_latents = []
                        for i in range(len(input_modals)):
                            if missing_code[i] == "1":  # 模态存在
                                combined_latents.append(source_latents[:, i : i + 1, :, :])
                            else:  # 模态缺失
                                combined_latents.append(current_latents[:, i : i + 1, :, :])

                        combined_latents = torch.cat(combined_latents, dim=1)

                        model_output = unet(
                            combined_latents,
                            timesteps=torch.tensor([t], device=device),
                            seg=seg,
                            sketch=sketch,
                            missing_code=missing_code_tensor,
                        )

                        # 执行去噪步骤
                        current_latents, _ = noise_scheduler.step(
                            model_output, t, current_latents
                        )

                        # 每隔N步保存中间结果
                        if t % 100 == 0 or t == noise_scheduler.timesteps[-1]:
                            # 按通道解码当前潜变量
                            current_image_list = []
                            for channel in range(current_latents.shape[1]):
                                channel_latents = current_latents[
                                    :, channel : channel + 1, :, :
                                ]
                                channel_image = autoencoder.decode_stage_2_outputs(
                                    channel_latents / scale_factor
                                )
                                current_image_list.append(channel_image)
                            current_image = torch.cat(current_image_list, dim=1)
                            denoising_chain.append(current_image.permute(0, 1, 3, 2).cpu())
                    final_image = current_image

                    # 找到缺失的模态索引
                    missing_indices = [
                        i for i, code in enumerate(missing_code) if code == "0"
                    ]
                    for missing_index in missing_indices:
                        target_images[0, missing_index].clamp_(0, 1)
                        final_image[0, missing_index].clamp_(0, 1)

                        # HWC
                        pred_image = visualize_2d_image(final_image[0, missing_index])
                        gt_images = visualize_2d_image(target_images[0, missing_index])

                        psnr_value = psnr(
                            pred_image[:, :, 0] / 255,
                            gt_images[:, :, 0] / 255,
                            data_range=1,
                        )
                        ssim_value = ssim(
                            pred_image[:, :, 0] / 255,
                            gt_images[:, :, 0] / 255,
                            data_range=1,
                        )
                        mae_value = mae(pred_image[:, :, 0] / 255, gt_images[:, :, 0] / 255)

                        missing_modal = all_modalities[missing_index]
                        print(
                            f"{batch[missing_modal][0].meta["filename_or_obj"]}: PSNR {psnr_value:.2f} dB, SSIM {ssim_value:.4f}, MAE {mae_value:.4f}"
                        )
                        results_dict[batch[missing_modal][0].meta["filename_or_obj"]] = {
                            "psnr": round(psnr_value, 2),
                            "ssim": round(ssim_value, 4),
                            "mae": round(mae_value, 4),
                        }

                        os.makedirs(
                            f"outputs/{missing_code_str}/test_latest/images/pred",
                            exist_ok=True,
                        )
                        PIL.Image.fromarray(pred_image.transpose((1, 0, 2))).save(
                            f'outputs/{missing_code_str}/test_latest/images/pred/{Path(batch[missing_modal][0].meta["filename_or_obj"]).stem}_pred.png'
                        )
                        os.makedirs(
                            f"outputs/{missing_code_str}/test_latest/images/target",
                            exist_ok=True,
                        )
                        PIL.Image.fromarray(gt_images.transpose((1, 0, 2))).save(
                            f'outputs/{missing_code_str}/test_latest/images/target/{Path(batch[missing_modal][0].meta["filename_or_obj"]).stem}_target.png'
                        )
