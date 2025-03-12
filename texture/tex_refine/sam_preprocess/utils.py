import os
import cv2
import numpy as np
import torch
from PIL import Image
from rembg import remove
from segment_anything import SamPredictor, sam_model_registry
import shutil

tracer_b7_ckpt_path = "/root/.cache/carvekit/checkpoints/tracer_b7/tracer_b7.pth"
fba_matting_path = "/root/.cache/carvekit/checkpoints/fba/fba_matting.pth"

if not os.path.exists(tracer_b7_ckpt_path):
    print("cp tracer_b7.pth to cache")
    os.makedirs(os.path.dirname(tracer_b7_ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(fba_matting_path), exist_ok=True)
    # shutil.copy("/aigc_cfs_gdp/neoshang/models/tracer_b7/tracer_b7.pth", tracer_b7_ckpt_path)
    # shutil.copy("/aigc_cfs_gdp/neoshang/models/fba_matting/fba_matting.pth", fba_matting_path)

    shutil.copy("/workspace/Tencent_XR_3DGen/tracer_b7/tracer_b7.pth", tracer_b7_ckpt_path)
    shutil.copy("/workspace/Tencent_XR_3DGen/fba/fba_matting.pth", fba_matting_path)

from carvekit.api.high import HiInterface

def sam_init(sam_checkpoint, device_id=0):
    model_type = "vit_h"

    device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
    print("[SAM] Loading sam to device ",device)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def check_sam_out(bbox_mask, image_height=512, image_width=512):
    minx = image_width / 4
    maxx = image_width / 4 * 3
    miny = image_height / 4
    maxy = image_height / 4 * 3

    miss_num = 0
    right_num = 0
    area_miss = 0
    area_right = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bbox_mask.astype(np.uint8) * 255)

    for label, stat, center in zip(labels[1:], stats[1:], centroids[1:]):
        if (center[0] < miny) or (center[0] > maxy) or \
            (center[1] < minx) or (center[1] > maxx):
            miss_num += 1
            area_miss += stat[-1]
        else:
            right_num += 1
            area_right += stat[-1]

    if area_right < area_miss:
        return False
    else:
        return True

def sam_segment(predictor, input_image, *bbox_sliders):
    print("enter sam")
    bbox = np.array(bbox_sliders)
    image = np.asarray(input_image)

    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox, multimask_output=True
    )
    image_h, image_w, _ = image.shape
    if not check_sam_out(masks_bbox[-1], image_height=image_h, image_width=image_w):
        print("check sam results failed, use rmbg module")
        image_out = to_white_bg(np.array(input_image), return_rgba=True) * 255
        image_out = np.clip(image_out, 0, 255)
        return Image.fromarray(image_out.astype(np.uint8), mode="RGBA")

    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = (
        masks_bbox[-1].astype(np.uint8) * 255
    )  # np.argmax(scores_bbox)

    print("leave sam")
    return Image.fromarray(out_image_bbox, mode="RGBA")


# contrast correction, rescale and recenter
def image_process(input_image, lower_contrast=False, rescale=True, bg_color=255, wh_ratio=0.8):
    image_arr = np.array(input_image)
    in_w, in_h = image_arr.shape[:2]
    if lower_contrast:
        alpha = 0.8  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        # Apply the contrast adjustment
        image_arr = cv2.convertScaleAbs(image_arr, alpha=alpha, beta=beta)
        image_arr[image_arr[..., -1] > 200, -1] = 255

    ret, mask = cv2.threshold(
        np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY
    )
    x, y, w, h = cv2.boundingRect(mask)
    np.set_printoptions(threshold=np.inf)
    mask_float = mask.astype(np.float32) / 255.0
    mask_float = np.concatenate([mask_float[..., None]]*3, axis=-1)
    white_bg_image = np.ones((in_w, in_h, 3), dtype=np.float32) * bg_color
    white_bg_image = image_arr[..., :-1] * mask_float + white_bg_image * (1-mask_float)

    max_size = max(w, h)
    ratio = wh_ratio
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_w
    padded_image = np.ones((side_len, side_len, 3), dtype=np.uint8) * bg_color
    center = side_len // 2

    padded_image[
        center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w
    ] = white_bg_image[y : y + h, x : x + w]

    mask_final = np.zeros((side_len, side_len), dtype=np.uint8)

    mask_final[
        center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w
    ] = mask[y : y + h, x : x + w]

    rgba = np.concatenate([padded_image, mask_final[..., None]], axis=-1)
    return rgba

class BackgroundRemoval:
    def __init__(self, device='cuda'):
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image

mask_predictor = BackgroundRemoval()

def remove_backgroud(images_array):
    images_array = np.transpose(images_array, (0, 2, 3, 1))
    images_array_out = []
    for image in images_array:
        with torch.no_grad():
            # image_rgb = resize_image(image_rgb, 320)
            image_rgba = mask_predictor(image)

        alpha = image_rgba[:, :, -1:] / 255.0
        images_rgb_array = np.array(image_rgba)
        image_out = np.clip(images_rgb_array * alpha + 255 * (1 - alpha), 0, 255)
        image_out = cv2.resize(image_out, (256, 256))
        images_array_out.append(image_out)
    return np.stack(images_array_out, axis=0).transpose(0, 3, 1, 2)

def get_mask_center(mask):
    x, y, w, h = cv2.boundingRect(mask)
    return int(x+w/2.0), int(y+h/2.0), w, h


def rescale_mask_img(image_out, origin_scale, new_scale, center_align=False):
    h, w, c = image_out.shape

    pad_h = int((h * origin_scale / new_scale - h) / 2)
    pad_w = int((w * origin_scale / new_scale - w) / 2)

    rgb = image_out[..., :3]
    mask = image_out[..., 3]
    rgb = np.pad(rgb, ((pad_h,pad_h),(pad_w,pad_w), (0,0)), mode='constant', constant_values=255.0)
    if not center_align:
        print("with out center_align")
        return rgb
    mask = np.pad(mask, ((pad_h,pad_h),(pad_w,pad_w)), mode='constant', constant_values=0.0)
    rgb_mask = np.concatenate([rgb, mask[..., None]], axis=-1)
    rgb_mask_resized = cv2.resize(rgb_mask, (w, h))

    x0, y0, _, _ = get_mask_center(image_out[..., -1])
    x1, y1, w_obj, h_obj = get_mask_center(rgb_mask_resized[..., -1])

    final_image = np.ones((h, w, 3)) * 255.0
    final_image_mask = np.zeros((h, w, 1))
    final_image_rgba = np.concatenate([final_image, final_image_mask], axis=-1)

    final_image_rgba[
        y0 - h_obj // 2 : y0 - h_obj // 2 + h_obj, x0 - w_obj // 2 : x0 - w_obj // 2 + w_obj
    ] = np.array(rgb_mask_resized)[y1 - h_obj // 2 : y1 - h_obj // 2 + h_obj, 
                                   x1 - w_obj // 2 : x1 - w_obj // 2 + w_obj]
    return final_image_rgba

def change_img_background_simple(images_array, rescale=False, origin_scale=0.9, new_scale=0.8, bg_color=255):
    channel_second = False
    if images_array.shape[1] == 3:
        channel_second = True
        images_array = np.transpose(images_array, (0, 2, 3, 1))
    images_array_out = []
    for image in images_array:
        image_out = to_white_bg(image, return_rgba=True, bg_color=bg_color) * 255
        if rescale:
            image_out = rescale_mask_img(image_out.clip(0, 255).astype(np.uint8), origin_scale, new_scale)
        image_out = image_out[..., :3]
        images_array_out.append(image_out)
    if channel_second:
        return np.stack(images_array_out, axis=0).transpose(0, 3, 1, 2)
    else:
        return np.stack(images_array_out, axis=0)


def to_white_bg(image, return_rgba=False, bg_color=255):
    with torch.no_grad():
        rgba = mask_predictor(image)
    if image.shape[0] > 512:
        # Erosion
        kernel = np.ones((3, 3), np.float32)
        rgba = rgba.astype(np.float32)/255.0
        rgba[:,:,-1] = cv2.erode(rgba[:,:,-1], kernel, iterations=3)
        rgba = rgba*255.0
        rgba = rgba.astype(np.uint8)
    elif (image.shape[0] > 256) and (image.shape[0] <= 512):
        # Erosion
        kernel = np.ones((3, 3), np.float32)
        rgba = rgba.astype(np.float32)/255.0
        rgba[:,:,-1] = cv2.erode(rgba[:,:,-1], kernel, iterations=1)
        rgba = rgba*255.0
        rgba = rgba.astype(np.uint8)
    elif image.shape[0] <= 256:
        # Erosion
        kernel = np.ones((3, 3), np.float32)
        rgba = rgba.astype(np.float32)/255.0
        rgba[:,:,-1] = cv2.erode(rgba[:,:,-1], kernel, iterations=0)
        rgba = rgba*255.0
        rgba = rgba.astype(np.uint8)
    rgb = rgba[:,:,:3].astype(np.float32)/255.0
    mask = rgba[:,:,-1].astype(np.float32)/255.0
    mask[mask>0.5] = 1.0
    mask[mask<=0.5] = 0.0
    rgb[:,:,0] = rgb[:,:,0]*mask+(1-mask)*bg_color/255.0
    rgb[:,:,1] = rgb[:,:,1]*mask+(1-mask)*bg_color/255.0
    rgb[:,:,2] = rgb[:,:,2]*mask+(1-mask)*bg_color/255.0
    rgb[rgb>1.0] = 1.0
    rgb[rgb<=0.0] = 0.0
    if return_rgba:
        return np.concatenate([rgb, mask[...,None]], axis=-1)
    else:
        return rgb



def pred_bbox(image):
    # image_nobg = remove(image.convert("RGBA"), alpha_matting=True)
    image_nobg = remove(image.convert("RGBA"))
    alpha = np.asarray(image_nobg)[:, :, -1]
    x_nonzero = np.nonzero(alpha.sum(axis=0))
    y_nonzero = np.nonzero(alpha.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    return x_min, y_min, x_max, y_max

def resize_image(input_raw, size):
    w, h = input_raw.size
    ratio = size / max(w, h)
    resized_w = int(w * ratio)
    resized_h = int(h * ratio)
    return input_raw.resize((resized_w, resized_h), Image.Resampling.LANCZOS)