import os
from natsort import natsorted, ns
import numpy as np
import torch


def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 0.1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def visualize_2d_image(image):
    """
    Prepare a 2D image for visualization.
    Args:
        image: image numpy array, sized (H, W)
    """
    # 判断输入是NumPy数组还是PyTorch张量
    if isinstance(image, torch.Tensor):
        # 如果是PyTorch张量，先转换为NumPy数组
        image_copy = image.cpu().numpy()
    elif isinstance(image, np.ndarray):
        # 如果是NumPy数组，直接复制
        image_copy = image.copy()
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    # draw image
    draw_img = normalize_image_to_uint8(image_copy)
    draw_img = np.stack([draw_img, draw_img, draw_img], axis=-1)
    return draw_img


def read_dir(dir_path, predicate=None, name_only=False, recursive=False):
    def is_matching_file(file_name):
        if predicate is None:
            return True
        elif isinstance(predicate, str):
            return (
                predicate == "dir" and os.path.isdir(os.path.join(dir_path, file_name))
            ) or (
                predicate == "file"
                and os.path.isfile(os.path.join(dir_path, file_name))
            )
        elif isinstance(predicate, list):
            return os.path.splitext(file_name)[-1][1:] in predicate
        elif callable(predicate):
            return predicate(file_name)
        return False

    # 检查目录是否存在
    if not os.path.isdir(dir_path):
        return []

    # 递归遍历目录
    output = []
    for f in natsorted(os.listdir(dir_path)):
        full_path = os.path.join(dir_path, f)
        if is_matching_file(f):
            output.append(f if name_only else full_path)
        if recursive and os.path.isdir(full_path):
            output.extend(read_dir(full_path, predicate, name_only, recursive))

    return natsorted(
        output, alg=ns.PATH
    )  # 要加alg=ns.PATH参数才和windows系统名称排序一致)
