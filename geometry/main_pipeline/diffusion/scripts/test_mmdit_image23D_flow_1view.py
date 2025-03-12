
import time
import numpy as np
import json
import os
import sys
import torch
import open_clip
from PIL import Image
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel, AutoImageProcessor, AutoModel
from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler, DDIMScheduler

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.vae.vae import get_vae_model
from models.vae.extract_geometry import extract_geometry
from utils.utils_obj import export_obj
from utils.rmbg1_4 import process_image_path, process_image_path_list
from models.diffusion.transformer_vector import SD3Transformer2DModel
from pipelines.pipeline_mmdit_flow import MMDitFlowPipeline

device = torch.device('cuda:0')


class unet_config:
    def __init__(self):
        pass


def test(args):
    exp_dir = args.exp_dir
    num_samples = args.num_samples

    if not args.save_dir:
        save_test_dir = os.path.join(
            exp_dir, "test_out_" + time.strftime('%Y-%m-%d-%H:%M:%S'))
    else:
        save_test_dir = os.path.join(
            args.save_dir, "test_out_" + time.strftime('%Y-%m-%d-%H:%M:%S'))

    configs_path = os.path.join(exp_dir, "train_configs.json")
    with open(configs_path, 'r') as fr:
        configs = json.load(fr)
    configs["exp_save_dir"] = exp_dir

    latent_sequence_num = configs["diffusion_config"]["latent_sequence_num"]
    condition_images_size = configs["dino_config"]["image_size"]

    weight_dtype = configs.get(
        "diffusion_config", {}).get("weight_dtype", "fp16")
    print("===============init info for clay diffusion=============\n")
    print(f"weight_dtype: {weight_dtype}")
    print("=======================================================\n")

    if weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # load vae
    vae = get_vae_model(configs)
    print("vae loading ckpt...")
    vae_state_dict = torch.load(
        configs["vae_config"]["pretrain_path"], map_location='cpu', weights_only=False)["state_dict"]
    new_vae_state_dict = {}
    for key, value in vae_state_dict.items():
        new_vae_state_dict[key[12:]] = value
    vae.load_state_dict(new_vae_state_dict, strict=True)
    del vae.encoder
    vae.to(device, dtype=torch.float16)
    vae.eval()
    print("vae_name:", configs["vae_config"]["vae_type"])

    # load dino model
    print("load dino from:", configs["dino_config"]["pretrain_dir"])
    dino_image_processor = AutoImageProcessor.from_pretrained(
        configs["dino_config"]["pretrain_dir"], local_files_only=True)
    dino_model = AutoModel.from_pretrained(configs["dino_config"]["pretrain_dir"], local_files_only=True).to(
        torch.device("cpu"), dtype=torch.float16)

    image_gray = Image.fromarray(
        (np.ones((512, 512, 3)) * 127).astype(np.uint8))

    def get_dino_feature(images_list):
        dino_model.to(device)
        image_conds = dino_image_processor(
            images_list, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**image_conds)
        last_hidden_states = outputs.last_hidden_state
        dino_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        return last_hidden_states

    # load clip model
    print("load clip from:", configs["clip_config"]["pretrain_dir"])
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-bigG-14', cache_dir=configs["clip_config"]["pretrain_dir"])
    clip_model.to(dtype=torch.float16)
    del clip_model.transformer
    del clip_model.vocab_size
    del clip_model.token_embedding
    del clip_model.positional_embedding
    del clip_model.ln_final
    del clip_model.text_projection
    del clip_model.attn_mask
    del clip_model.logit_scale
    clip_image_processor = AutoImageProcessor.from_pretrained(
        configs["clip_config"]["pretrain_dir"], local_files_only=True)

    def get_clip_feature(images_list):
        clip_model.to(device)
        image_conds = clip_image_processor(
            images_list, return_tensors="pt").to(device)["pixel_values"]

        with torch.no_grad():
            image_latents_clip = clip_model.encode_image(
                image_conds.to(dtype=torch.float16))
        image_latents_clip = image_latents_clip.to(
            dtype=weight_dtype).detach().contiguous()

        clip_model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        return image_latents_clip

    dino_empty_local_path = configs["dino_config"]["gray_dino_feature_path"]
    clip_empty_global_path = configs["clip_config"]["gray_clip_feature_path"]
    gray_image_local_embedding = torch.load(
        dino_empty_local_path, map_location="cpu")
    gray_image_global_embedding = torch.load(
        clip_empty_global_path, map_location="cpu")
    print(
        f"gray_image_local_embedding shape: {gray_image_local_embedding.shape}")
    print(
        f"gray_image_global_embedding shape: {gray_image_global_embedding.shape}")

    # diffusion pipeline
    diffusion_model_config = configs["diffusion_config"]
    unet_config_path = os.path.join(args.exp_dir, "unet/config.json")

    dirs = os.listdir(args.exp_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None
    if path is not None:
        ckpt_dir = os.path.join(args.exp_dir, path, "unet_ema")
        if not os.path.exists(ckpt_dir):
            ckpt_dir = os.path.join(args.exp_dir, path, "unet")
    else:
        ckpt_dir = os.path.join(args.exp_dir, "unet")
    print(f"load ckpt from {ckpt_dir}")
    ckpt_name = os.path.basename(path)
    save_test_dir = save_test_dir + "_" + ckpt_name
    print(f"save_test_dir: {save_test_dir}")
    os.makedirs(save_test_dir, exist_ok=True)


    # sd3
    unet = SD3Transformer2DModel(**diffusion_model_config)
    unet.load_checkpoint(ckpt_dir)
    unet.eval()

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        exp_dir, subfolder="scheduler")

    if args.model_type == "sd3":
        generator = MMDitFlowPipeline(
            transformer=unet,
            scheduler=noise_scheduler,
        )

    generator.to(torch.device("cpu"), dtype=weight_dtype)

    image_dir = args.image_dir
    image_path_list = [os.path.join(image_dir, x)
                       for x in os.listdir(image_dir)]
    image_path_list.sort()
    print(f"image_dir_list num: {len(image_path_list)}")

    num = 0
    for i, image_path in enumerate(image_path_list):
        if i >= num_samples:
            break

        # save image
        image_name = os.path.basename(image_path)
        image_base_name = image_name.split(".")[0]
        classname = "test_images"
        objname = image_base_name
        mesh_name = os.path.join(save_test_dir,  classname, objname)
        if os.path.exists(mesh_name):
            continue
        os.makedirs(mesh_name, exist_ok=True)

        # test diffusion
        try:
            image, mask = process_image_path(
                image_path, bg_color=127, wh_ratio=0.9, rmbg_type="1.4")
        except:
            continue

        image_save_path = os.path.join(mesh_name, os.path.basename(image_path))
        os.system(f"cp \"{image_path}\" \"{image_save_path}\"")

        image_latents_local_dino = get_dino_feature([image])
        image_latents_local_dino = image_latents_local_dino.to(
            dtype=weight_dtype).contiguous()

        image_latents_global_clip = get_clip_feature([image])
        image_latents_global_clip = image_latents_global_clip.to(
            dtype=weight_dtype).contiguous()

        generator.to(device)

        with torch.autocast("cuda"):
            latents_gen = generator(sequence_length=latent_sequence_num,
                                    feature_dim=64,
                                    prompt_embeds=image_latents_local_dino,
                                    pooled_prompt_embeds=image_latents_global_clip,
                                    negative_prompt_embeds=gray_image_local_embedding,
                                    negative_pooled_prompt_embeds=gray_image_global_embedding,
                                    num_inference_steps=75,
                                    guidance_scale=7.5)
        generator.to("cpu")

        classname = "test_images"
        objname = os.path.basename(image_path)
        torch.cuda.empty_cache()

        latents = vae.decode(latents_gen.to(torch.float16)
                             )  # [B, num_latents, width]

        box_v = 1.05
        try:
            mesh_outputs, _, vertices_sparse, faces_sparse = extract_geometry(
                vae,
                latents,
                bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
                octree_depth=8,
                method='sparse'
            )
        except:
            print(f"extract_geometry failed: {image_path}")

        # save mesh
        os.makedirs(mesh_name, exist_ok=True)
        export_obj(vertices_sparse,
                   faces_sparse[:, ::-1], os.path.join(mesh_name, "mesh.obj"))

        num += 1
        print(f"generated {num}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'train_configs.json' and 'test_configs.json'",
    )
    arg_parser.add_argument(
        "--image_dir", default=None,
        help="specify the test image_dir",
    )

    arg_parser.add_argument(
        "--save_dir", default=None,
        help="specify the save_dir",
    )

    arg_parser.add_argument(
        '--generate_obj', action="store_true", default=False)

    arg_parser.add_argument("--num_samples", "-n", default=10000, type=int,
                            help='number of samples to generate and reconstruct')
    arg_parser.add_argument(
        "--model_type", default="sd3", type=str,
        help="dit, craftsman, sd3"
    )
    args = arg_parser.parse_args()

    test(args)
