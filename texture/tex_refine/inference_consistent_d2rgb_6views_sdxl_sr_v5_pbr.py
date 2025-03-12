import os
import sys
import time
from pathlib import Path
from dotmap import DotMap
from datetime import datetime
import torch
import torch.nn.functional as F
import requests
from PIL import Image, ImageEnhance 
import random
import numpy as np
import argparse
from pdb import set_trace as st
import nvdiffrast.torch as dr
import gc
import openai

# Pre-process for multi-view d2rgb
from sam_preprocess.run_sam import process_image_path

# VAE & ControlNet
from diffusers import AutoencoderKL, ControlNetModel
# Pipeline
from d2rgb_pipeline_6views import Zero123PlusPipeline
from d2rgb_pipeline_6views_3views import Zero123PlusPipeline as Zero123PlusPipeline_3views
from sdxl_pipeline import StableDiffusionXLControlNetImg2ImgPipeline
# Scheduler
from consistent_scheduler_eular_ancestral_discrete_newbaking_v2_2stage_90s import ConsistentEulerAncestralDiscreteScheduler
from consistent_scheduler_eular_ancestral_discrete_newbaking_6views_v2_2stage_90s import ConsistentEulerAncestralDiscreteScheduler as ConsistentEulerAncestralDiscreteScheduler_6views_d2rgb
from consistent_scheduler_eular_ancestral_discrete_newbaking_v5_pbr import ConsistentEulerAncestralDiscreteScheduler as ConsistentEulerAncestralDiscreteScheduler_6views
# Real-ESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
# Prompt weighting
from compel import Compel, ReturnedEmbeddingsType

# Captioning
# from gpt_caption import gpt_caption

import base64
from io import BytesIO

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

def gpt_caption(ref_img_path, retries=2, default=""):
    """
    Generates a caption for the given image using OpenAI's GPT-4 Vision API.

    Parameters:
        ref_img_path (str): The file path to the reference image.
        retries (int): The number of times to retry the API call in case of failure.
        default (str): The default caption to return if all retries fail.

    Returns:
        str: The generated caption or the default caption if an error occurs.
    """
    # Load and encode the image in base64
    try:
        with Image.open(ref_img_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error loading or encoding image: {e}")
        return default

    # Define the system prompt
    system_prompt = "You are an AI assistant that generates descriptive captions for images."

    # Attempt to get a caption from the API, retrying if necessary
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "image", "image": img_base64}]}
                ]
            )
            caption = response['choices'][0]['message']['content'].strip()
            return caption
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

    # If all retries fail, return the default caption
    return default


def init_models(
    rotate_input_mesh=False,
    three_views_input=False,
    debug_outputs=True
):


    # args
    args = DotMap()

    args.debug_outputs = debug_outputs

    # Input args
    args.rotate_input_mesh = rotate_input_mesh
    args.three_views_input = three_views_input

    ##### D2rgb #####
    if three_views_input:
        args.d2rgb_num_inference_steps = 75
        args.d2rgb_cfg_scheduler_type = 'linear'
        args.d2rgb_cfg_scheduler_params = [3.6,3.5]
        # args.d2rgb_perview_cfg_weight = [1.0, 1.2, 1.3, 1.2, 1.2, 1.2]
        args.d2rgb_perview_cfg_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        args.d2rgb_image_sharpen_factor = 1.5
        args.d2rgb_disable_consistency = False
        args.d2rgb_apply_consistent_interval = 15
    else: # single view input
        args.d2rgb_num_inference_steps = 50
        args.d2rgb_cfg_scheduler_type = 'linear'
        args.d2rgb_cfg_scheduler_params = [7.5,3]
        # args.d2rgb_perview_cfg_weight = [1.0, 1.2, 1.3, 1.2, 1.2, 1.2]
        args.d2rgb_perview_cfg_weight = [1.0, 1.1, 1.1, 1.1, 1.0, 1.0]
        args.d2rgb_image_sharpen_factor = 4.0
        args.d2rgb_disable_consistency = False
        args.d2rgb_apply_consistent_interval = 11

    args.d2rgb_texture_map_resolution = 3072

    ##### PBR #####


    ##### SDXL #####
    args.pretrained_model_name_or_path="./weights/helloworldXL70/"
    # args.pretrained_model_name_or_path="./pretrained/sdxl-lightning-4steps/"
    # args.pretrained_model_name_or_path = '/aigc_cfs_gdp/jiayu/consistent_scheduler_xibin/pretrained/stable-diffusion-xl-base-1.0'
    
    args.resolution = 1280
    if three_views_input:
        # 3 ref images.
        args.num_inference_steps = 16
        args.sdxl_denoising_strength = 0.3
        args.sdxl_cfg_scheduler_params = [24,2]
        args.sdxl_1st_stage_output_sharpen_factor = 1.0

        args.xyz_controlnet_path = "./weights/pretrained_sdxl_xyz_controlnet"
        args.sdxl_controlnet_scale_xyz = 0.0
        args.sdxl_controlnet_scale_tile = 0.6
        args.sdxl_controlnet_scale_depth = 0.9
        args.sdxl_ip_adapter_scale = 1.5
    else:
        # args.num_inference_steps = 16
        # args.sdxl_denoising_strength = 0.7
        # args.sdxl_cfg_scheduler_params = [24,2]
        # args.sdxl_1st_stage_output_sharpen_factor = 3.5
        args.num_inference_steps = 16
        args.sdxl_denoising_strength = 0.55
        args.sdxl_cfg_scheduler_params = [24,2]
        args.sdxl_1st_stage_output_sharpen_factor = 1.3

        args.xyz_controlnet_path = "./weights/pretrained_sdxl_xyz_controlnet"
        args.sdxl_controlnet_scale_xyz = 1.0
        args.sdxl_controlnet_scale_tile = 0.6
        args.sdxl_controlnet_scale_depth = 0.9
        args.sdxl_ip_adapter_scale = 1.5

    args.sdxl_lora_stack = [
        "XDetail_light",
        # "contrast_tool",
        # "sdxl_lightning_8step_lora",
        # "xl_color_temp",
    ]
    args.sdxl_lora_stack_weights = [
        2,
        # 0.8,
        # 0.7,
        # -1.0
    ]
    

    args.sdxl_texture_map_resolution = args.d2rgb_texture_map_resolution

    args.sdxl_apply_consistent_interval = 3
    
    ##### SDXL second refine #####
    args.sdxl_second_refine_lora_stack = [
        # "XDetail_light",
        # "contrast_tool",
        # "sdxl_lightning_8step_lora",
        # "xl_color_temp",
    ]
    args.sdxl_second_refine_lora_stack_weights = [
        # 3.0,
        # 0.8,
        # 0.7,
        # -1.0
    ]

    args.sdxl_second_refine_resolution = 1280
    # args.sdxl_second_refine_resolution = 2368
    args.sdxl_second_refine_num_inference_steps = 12
    args.sdxl_second_refine_denoising_strength = 0.3
    args.sdxl_second_refine_cfg_scheduler_params = [24,1]
    
    args.sdxl_second_refine_controlnet_scale_xyz = 0.0
    args.sdxl_second_refine_controlnet_scale_tile = 0.0
    args.sdxl_second_refine_controlnet_scale_depth = 0.9
    args.sdxl_second_refine_ip_adapter_scale = 1.5

    args.sdxl_second_refine_disable_consistency = False
    args.sdxl_second_refine_apply_consistent_interval = 4

    # General
    args.enable_xformers_memory_efficient_attention = True
    args.renderer_extra_scale_up_factor = 2
    weight_dtype = torch.float16

    #################################### REAL-ESRGAN ####################################
    # restorer
    print("[REAL-ESRGAN] Loading model...")
    # Small model
    denoise_strength = 0
    model_path='weights/realesrgan/realesr-general-x4v3.pth'
    # model_path='pretrained/realesrgan/realesr-general-wdn-x4v3.pth'
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    # wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
    # model_path = [model_path, wdn_model_path]
    # dni_weight = [denoise_strength, 1 - denoise_strength]
    upsampler_small = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0)

    # Large x2 model
    model_path='weights/realesrgan/RealESRGAN_x2plus.pth'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    netscale = 2
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0)

    # Large x4 model
    # model_path='pretrained/realesrgan/RealESRGAN_x4plus.pth'
    # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    # netscale = 4
    # upsampler = RealESRGANer(
    #     scale=netscale,
    #     model_path=model_path,
    #     dni_weight=None,
    #     model=model,
    #     tile=0,
    #     tile_pad=10,
    #     pre_pad=0,
    #     half=True,
    #     gpu_id=0)


    #################################### D2RGB ####################################
    # Create pipeline (d2rgb)
    if args.three_views_input:
        args.z123_model_path = "./weights/zero23plus_v25_4vews_abs_39000" # 3views model
        args.controlnet_model_path = "./weights/zero23_controlnet_8000"

        pipeline = Zero123PlusPipeline_3views.from_pretrained(
            args.z123_model_path, torch_dtype=torch.float16
        )

        controlnet_d2rgb = ControlNetModel.from_pretrained(
            args.controlnet_model_path, torch_dtype=torch.float16
            )

        # 3view input d2rgb use 4view consistent scheduler.
        pipeline.scheduler = ConsistentEulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')

    else:
        # z123_model_path = "/aigc_cfs/xibinsong/models/zero123plus_v28_1cond_6views_same_azim_512" # 6views model
        # # controlnet_model_path = "/aigc_cfs_4/xibin/code/diffusers_triplane_6views_non_rotate/configs/zero123plus/zero123plus_v28_1cond_6views_same_azim_512_controlnet_v100/checkpoint-14000/controlnet" # Copied file from above
        # controlnet_model_path = "/aigc_cfs_gdp/jiayu/consistent_scheduler_xibin/pretrained/zero123plus_v28_1cond_6views_same_azim_512_controlnet_v100_checkpoint_14000_controlnet" # Copied file from above

        # New unified model
        args.z123_model_path = "./weights/zero123plus_v29_1cond_6views_090180270"
        # controlnet_model_path = "/aigc_cfs_gdp/jiayu/consistent_scheduler_xibin/pretrained/zero123plus_v29_1cond_6views_090180270_controlnet_checkpoint-6000_controlnet"
        args.controlnet_model_path = "./weights/zero123plus_v29_1cond_6views_090180270_controlnet_15000"

        pipeline = Zero123PlusPipeline.from_pretrained(
            args.z123_model_path, torch_dtype=torch.float16
        )
        
        controlnet_d2rgb = ControlNetModel.from_pretrained(
            args.controlnet_model_path, torch_dtype=torch.float16
            )

        pipeline.scheduler = ConsistentEulerAncestralDiscreteScheduler_6views_d2rgb.from_config(pipeline.scheduler.config, timestep_spacing='trailing')

    # Feel free to tune the scheduler
    # pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    #         pipeline.scheduler.config, timestep_spacing='trailing'
    #     )
    pipeline.scheduler.vae = pipeline.vae
    pipeline.scheduler.upsampler = upsampler # Real-ESRGAN
    pipeline = pipeline.to("cuda")

    # cfg scheduler
    pipeline.cfg_scheduler_type = args.d2rgb_cfg_scheduler_type
    pipeline.cfg_scheduler_params = args.d2rgb_cfg_scheduler_params
    # pipeline.perview_cfg_weight = [1.1, 1.5, 2.2, 1.5, 1.5, 1.5]
    pipeline.perview_cfg_weight = args.d2rgb_perview_cfg_weight

    # Enable xformers
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    d2rgb_pipeline = pipeline

    #################################### PBR ####################################
    from diffusers import EulerAncestralDiscreteScheduler
    from diffusers_albedo_metallic_roughness.src.diffusers.pipelines.img2img.pipeline_img2img import Img2ImgPipeline as PBRPipeline

    # Albedo model
    albedo_pipeline = PBRPipeline.from_pretrained(
                    "./weights/img2img_albedo_6views_512_250k",
                    torch_dtype=torch.float16,
                    local_files_only=True
                )
    # Feel free to tune the scheduler
    albedo_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        albedo_pipeline.scheduler.config,
        timestep_spacing='trailing'
    )
    albedo_pipeline.to('cuda')

    # Metallic Roughness model
    metallic_roughness_pipeline = PBRPipeline.from_pretrained(
                    "./weights/img2img_material_two_outputs",
                    torch_dtype=torch.float16,
                    local_files_only=True
                )

    # Feel free to tune the scheduler
    metallic_roughness_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        metallic_roughness_pipeline.scheduler.config,
        timestep_spacing='trailing'
    )
    metallic_roughness_pipeline.to('cuda')

    # Reference view albedo model
    from diffusers_albedo_metallic_roughness.src.diffusers.pipelines.img2img.pipeline_img2img_single import Img2ImgPipeline as SingleAlbedoPipeline
    ref_albedo_pipeline = SingleAlbedoPipeline.from_pretrained(
                        "./weights/delight_models_full_light",
                        torch_dtype=torch.float16,
                        local_files_only=True
                    )

    # Feel free to tune the scheduler
    ref_albedo_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        ref_albedo_pipeline.scheduler.config,
        timestep_spacing='trailing'
    )
    ref_albedo_pipeline.to('cuda')

    # breakpoint()

    #################################### SDXL ####################################
    # Create pipeline (SDXL)
    print("Creating pipeline...")
    print("vae...")
    vae = AutoencoderKL.from_pretrained("weights/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    vae.enable_slicing() # Save memory...
    vae.enable_tiling() # Save memory...
    print("controlnet...")
    # XYZ controlnet
    controlnet_xyz = ControlNetModel.from_pretrained(args.xyz_controlnet_path, torch_dtype=weight_dtype)
    # tile controlnet
    controlnet_tile = ControlNetModel.from_pretrained('./weights/TTPLANET_Controlnet_Tile/', torch_dtype=weight_dtype)
    # depth controlnet
    controlnet_depth = ControlNetModel.from_pretrained('./weights/sdxl_depth_controlnet_fp16/', torch_dtype=weight_dtype)

    print("pipeline...")
    pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=[controlnet_xyz,controlnet_tile,controlnet_depth],
        # controlnet=[controlnet_tile,controlnet_depth],
        torch_dtype=weight_dtype,
        vae=vae,
    )
    pipeline.scheduler = ConsistentEulerAncestralDiscreteScheduler_6views.from_config(pipeline.scheduler.config)
    pipeline.scheduler.vae = vae
    pipeline.scheduler.upsampler = upsampler # Real-ESRGAN
    pipeline.scheduler.upsampler_small = upsampler_small # Real-ESRGAN small model, for larger upscaling
    # pipeline.vae = pipeline.vae.to(torch.float32)

    # cfg scheduler for SDXL
    pipeline.cfg_scheduler_type = 'linear'
    pipeline.cfg_scheduler_params = args.sdxl_cfg_scheduler_params

    # Load the LoRA
    print("lora...")
    if "XDetail_light" in args.sdxl_lora_stack:
        pipeline.load_lora_weights('./weights/', weight_name='XDetail_light.safetensors', adapter_name="XDetail_light")
    if "contrast_tool" in args.sdxl_lora_stack:
        pipeline.load_lora_weights('./weights/', weight_name='SDS_Contrast tool_XL.safetensors', adapter_name="contrast_tool")
    if "sdxl_lightning_8step_lora" in args.sdxl_lora_stack:
        pipeline.load_lora_weights('./weights/', weight_name='sdxl_lightning_8step_lora.safetensors', adapter_name="sdxl_lightning_8step_lora")
    if "xl_color_temp" in args.sdxl_lora_stack:
        pipeline.load_lora_weights('./weights/', weight_name='xl_color_temp.safetensors', adapter_name="xl_color_temp")
    # Activate the LoRA
    # pipeline.set_adapters(["extremely detailed"], adapter_weights=[2.0])
    # pipeline.set_adapters(["XDetail_light","contrast_tool","sdxl_lightning_8step_lora","xl_color_temp"], adapter_weights=[4.0,1.0,0.8,-1.0])
    # pipeline.set_adapters(["XDetail_light","contrast_tool","sdxl_lightning_8step_lora","xl_color_temp"], adapter_weights=[4.0,0.8,0.7,-1.0])
    # pipeline.set_adapters(["XDetail_light","contrast_tool","sdxl_lightning_8step_lora","xl_color_temp"], adapter_weights=[3.0,0.6,0.6,-0.7])
    if len(args.sdxl_lora_stack) > 0:
        pipeline.set_adapters(args.sdxl_lora_stack, adapter_weights=args.sdxl_lora_stack_weights)

    pipeline = pipeline.to("cuda")

    # Enable xformers
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    sdxl_pipeline = pipeline

    # IPAdapter
    # from ip_adapter import IPAdapterXL
    # image_encoder_path = "/aigc_cfs_gdp/jiayu/comfyui_models_2/ipadapter/image_encoder"
    # # ip_ckpt = "/aigc_cfs_gdp/jiayu/comfyui_models_2/ipadapter/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
    # ip_ckpt = "/aigc_cfs_gdp/jiayu/comfyui_models_2/ipadapter/ip-adapter_sdxl.bin"
    # # load ip-adapter
    # ip_model = IPAdapterXL(sdxl_pipeline, image_encoder_path, ip_ckpt, device='cuda')

    # Better IPAdapter
    from ip_adapter import IPAdapterPlusXL
    image_encoder_path = "./weights/IP-Adapter_image_encoder"
    ip_ckpt = "./weights/ip-adapter-plus_sdxl_vit-h.bin"
    # load ip-adapter
    ip_model = IPAdapterPlusXL(sdxl_pipeline, image_encoder_path, ip_ckpt, device='cuda', num_tokens=16)

    # Prompt weighting function
    compel = Compel(
        tokenizer=[sdxl_pipeline.tokenizer, sdxl_pipeline.tokenizer_2], 
        text_encoder=[sdxl_pipeline.text_encoder, sdxl_pipeline.text_encoder_2], 
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
        requires_pooled=[False, True], 
        truncate_long_prompts=False
    )

    # global_ctx = dr.RasterizeCudaContext()

    #################################### Collect all models ####################################
    models = {
        'upsampler_small':upsampler_small,
        'upsampler':upsampler,
        'd2rgb_pipeline':d2rgb_pipeline,
        'controlnet_d2rgb':controlnet_d2rgb,
        'albedo_pipeline':albedo_pipeline,
        'metallic_roughness_pipeline':metallic_roughness_pipeline,
        'ref_albedo_pipeline':ref_albedo_pipeline,
        'sdxl_pipeline':sdxl_pipeline,
        'ip_model':ip_model,
        'compel':compel,
        'args':args,
        # 'global_ctx':global_ctx,

    }

    torch.cuda.empty_cache()

    return models

@torch.no_grad()
def run_d2rgb_sdxl_sr(
    models,
    obj_path,  
    img_path,
    out_dir,
    job_id,
    seed=None):

    start_time = time.perf_counter()

    # Unpack models
    upsampler_small = models['upsampler_small']
    upsampler = models['upsampler']
    d2rgb_pipeline = models['d2rgb_pipeline']
    controlnet_d2rgb = models['controlnet_d2rgb']
    albedo_pipeline = models['albedo_pipeline']
    metallic_roughness_pipeline = models['metallic_roughness_pipeline']
    ref_albedo_pipeline = models['ref_albedo_pipeline']
    sdxl_pipeline = models['sdxl_pipeline']
    ip_model = models['ip_model']
    compel = models['compel']
    args = models['args']
    # global_ctx = models['global_ctx']

    # Initialize global_ctx
    global_ctx = dr.RasterizeCudaContext()

    weight_dtype = torch.float16
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = torch.Generator(device="cuda")

    # Input files
    print("\n[DEBUG] img_path:", img_path, "\n")
    ref_img_path = img_path
    obj_path = obj_path

    prompt_file = os.path.join(os.path.dirname(ref_img_path),"prompt.txt")
    if not os.path.exists(prompt_file):
        prompt_file = None

    #################################### RUN D2RGB ####################################
    # Re initialize d2rgb pipeline for fucking sake
    if args.three_views_input:
        print('[D2RGB] Reinitialize d2rgb pipeline......')
        # z123_model_path = "/aigc_cfs_gdp/xibin/z123_control/models/3view_models/zero23plus_v25_4vews_abs_39000" # 3views model
        # controlnet_model_path = "/aigc_cfs_gdp/xibin/z123_control/models/3view_models/controlnet-8000/controlnet"

        d2rgb_pipeline = Zero123PlusPipeline_3views.from_pretrained(
            args.z123_model_path, torch_dtype=torch.float16
        )
        d2rgb_pipeline.scheduler = ConsistentEulerAncestralDiscreteScheduler.from_config(d2rgb_pipeline.scheduler.config, timestep_spacing='trailing')
        d2rgb_pipeline.scheduler.vae = d2rgb_pipeline.vae
        d2rgb_pipeline.scheduler.upsampler = upsampler # Real-ESRGAN
        d2rgb_pipeline = d2rgb_pipeline.to("cuda")

        # cfg scheduler
        d2rgb_pipeline.cfg_scheduler_type = args.d2rgb_cfg_scheduler_type
        d2rgb_pipeline.cfg_scheduler_params = args.d2rgb_cfg_scheduler_params
        # pipeline.perview_cfg_weight = [1.1, 1.5, 2.2, 1.5, 1.5, 1.5]
        d2rgb_pipeline.perview_cfg_weight = args.d2rgb_perview_cfg_weight

        # Enable xformers
        if args.enable_xformers_memory_efficient_attention:
            d2rgb_pipeline.enable_xformers_memory_efficient_attention()

        models['d2rgb_pipeline'] = d2rgb_pipeline

    ref_img_file = ref_img_path
    # if prompt_file is not None:
    #     prompt = open(prompt_file, 'r', encoding='utf-8').read()
    # else:
    #     prompt = ''
    # D2rgb does not need prompt
    prompt = ''
    mesh = obj_path
    print(f"Prompt: {prompt}, Mesh: {mesh}")
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join("./outputs_service_d2rgb_6views_sdxl_sr_local",current_time[:10],current_time+"_"+str(job_id))
    # output_path = os.path.join("/aigc_cfs_gdp/jiayu/tmp/outputs_baking_d2rgb",mesh.split("/")[5])
    os.makedirs(output_path, exist_ok=True)
    # Remove files in output_path
    # for filename in os.listdir(output_path):
    #     file_path = os.path.join(output_path, filename)
    #     try:
    #         if os.path.isfile(file_path) or os.path.islink(file_path):
    #             os.unlink(file_path)
    #         elif os.path.isdir(file_path):
    #             shutil.rmtree(file_path)
    #     except Exception as e:
    #         print(f'Failed to delete {file_path}. Reason: {e}')

    # Initialize renderer in consistent scheduler
    obj_path = mesh
    if args.three_views_input == False:
        d2rgb_per_view_image_size = 512
    else:
        d2rgb_per_view_image_size = 416
    render_resolution = int(d2rgb_per_view_image_size*args.renderer_extra_scale_up_factor)
    print("Render resolution", render_resolution)
    print("SDXL per view resolution", d2rgb_per_view_image_size)
    scale_factor = 0.9 # for tripo
    unwrap_uv = False
    d2rgb_pipeline.scheduler.init_renderer(
        mesh=obj_path, 
        render_size=render_resolution, 
        scale_factor=scale_factor, 
        renderer_extra_scale_up_factor=args.renderer_extra_scale_up_factor, 
        d2rgb_special_scale=True,
        texture_resolution=args.d2rgb_texture_map_resolution,
        num_inference_steps=args.num_inference_steps,
        rotate_input_mesh=args.rotate_input_mesh,
        ctx=global_ctx,
        unwrap_uv=unwrap_uv
    )
    input_has_uv = d2rgb_pipeline.scheduler.input_has_uv
    if unwrap_uv:
        input_has_uv = False
    print("[NVDIFFRAST] Using global ctx:", global_ctx)
    # Specify paths for consistent scheduler
    d2rgb_pipeline.scheduler.mesh = mesh # Path to obj file
    d2rgb_pipeline.scheduler.output_path = output_path

    d2rgb_pipeline.scheduler.disable_consistency = args.d2rgb_disable_consistency
    d2rgb_pipeline.scheduler.apply_consistent_interval = args.d2rgb_apply_consistent_interval

    # Get xyz map from renderer, it's already computed when initializing the renderer
    xyz_maps = d2rgb_pipeline.scheduler.xyz
    xyz_maps = F.interpolate(xyz_maps.permute(0,3,1,2), size=(d2rgb_per_view_image_size, d2rgb_per_view_image_size), mode='nearest').permute(0,2,3,1).contiguous()

    # Run single view or multi-view d2rgb model
    if args.three_views_input == False: # Single view model

        xyz_maps_grid = torch.zeros([d2rgb_per_view_image_size*3,d2rgb_per_view_image_size*2,3],dtype=torch.float32,device='cuda')
        xyz_maps_grid[:d2rgb_per_view_image_size,:d2rgb_per_view_image_size] = xyz_maps[0] # 1
        xyz_maps_grid[:d2rgb_per_view_image_size,d2rgb_per_view_image_size:] = xyz_maps[1] #    2
        xyz_maps_grid[d2rgb_per_view_image_size:2*d2rgb_per_view_image_size,:d2rgb_per_view_image_size] = xyz_maps[2] # 3
        xyz_maps_grid[d2rgb_per_view_image_size:2*d2rgb_per_view_image_size:,d2rgb_per_view_image_size:] = xyz_maps[3] #    4
        xyz_maps_grid[2*d2rgb_per_view_image_size:,:d2rgb_per_view_image_size] = xyz_maps[4] # 5
        xyz_maps_grid[2*d2rgb_per_view_image_size:,d2rgb_per_view_image_size:] = xyz_maps[5] #    6

        # Swap axis
        xyz_maps_grid = torch.stack([xyz_maps_grid[:,:,0],-xyz_maps_grid[:,:,2],xyz_maps_grid[:,:,1]],dim=-1)
        # plt.imsave("debug.png",(xyz_maps_grid.data.cpu().numpy()+1)/2)
        # 127
        background_mask = (xyz_maps_grid[:,:,0] == 0)
        # xyz_maps_grid[background_mask] = 0
        xyz_control_image = Image.fromarray(((xyz_maps_grid.data.cpu().numpy()+1)/2*255).astype(np.uint8))
        if args.debug_outputs:
            xyz_control_image.save(os.path.join(output_path, 'control_image_xyz_d2rgb.png'))

        # Replace ref_img with the gen3d processed one, if exists.
        gen3d_img_file = os.path.join(os.path.dirname(ref_img_file), 'gen3d', '0.png')
        if os.path.exists(gen3d_img_file):
            print("[D2RGB] Reference image replaced by ", gen3d_img_file)
            ref_img_file = gen3d_img_file
        else:
            print("[D2RGB] Reference image not replaced", gen3d_img_file)

        ref_img = Image.open(ref_img_file).convert("RGB")
        if args.debug_outputs:
            ref_img.save(os.path.join(output_path, 'ref_img.png'))
        cond = ref_img

    else: # Multi-view drgb model

        xyz_maps_grid = torch.zeros([d2rgb_per_view_image_size*2,d2rgb_per_view_image_size*2,3],dtype=torch.float32,device='cuda')
        xyz_maps_grid[:d2rgb_per_view_image_size,:d2rgb_per_view_image_size] = xyz_maps[0] # 1
        xyz_maps_grid[:d2rgb_per_view_image_size,d2rgb_per_view_image_size:] = xyz_maps[1] #    2
        xyz_maps_grid[d2rgb_per_view_image_size:,:d2rgb_per_view_image_size] = xyz_maps[2] # 3
        xyz_maps_grid[d2rgb_per_view_image_size:,d2rgb_per_view_image_size:] = xyz_maps[3] #    4

        # Swap axis
        xyz_maps_grid = torch.stack([xyz_maps_grid[:,:,0],-xyz_maps_grid[:,:,2],xyz_maps_grid[:,:,1]],dim=-1)
        # plt.imsave("debug.png",(xyz_maps_grid.data.cpu().numpy()+1)/2)
        # 127
        background_mask = (xyz_maps_grid[:,:,0] == 0)
        # xyz_maps_grid[background_mask] = 0
        xyz_control_image = Image.fromarray(((xyz_maps_grid.data.cpu().numpy()+1)/2*255).astype(np.uint8))
        if args.debug_outputs:
            xyz_control_image.save(os.path.join(output_path, 'control_image_xyz_d2rgb.png'))
        
        parent_path = os.path.dirname(img_path)
        input_img_0 = os.path.join(parent_path, "input_img_0.png")
        input_img_1 = os.path.join(parent_path, "input_img_1.png")
        input_img_2 = os.path.join(parent_path, "input_img_2.png")
        input_img_3 = os.path.join(parent_path, "input_img_3.png")
        input_img_0_sr = os.path.join(parent_path, "output_0.png")
        input_img_1_sr = os.path.join(parent_path, "output_1.png")
        input_img_2_sr = os.path.join(parent_path, "output_2.png")
        input_img_3_sr = os.path.join(parent_path, "output_3.png")

        image_path_list = [] 

        num_sr_imgs = 0
        num_imgs = 0

        if os.path.exists(input_img_0_sr):
            print("input sr image 0 exist!")
            image_path_list.append(input_img_0_sr)
            num_sr_imgs += 1
        else:
            print("input sr image 0 not exist!")

        if os.path.exists(input_img_1_sr):
            print("input sr image 1 exist!")
            image_path_list.append(input_img_1_sr)
            num_sr_imgs += 1
        else:
            print("input sr image 1 not exist!")
            
        if os.path.exists(input_img_2_sr):
            print("input sr image 2 exist!")
            image_path_list.append(input_img_2_sr)
            num_sr_imgs += 1
        else:
            print("input sr image 2 not exist!")
        
        if os.path.exists(input_img_3_sr):
            print("input image 3 exist!")
            image_path_list.append(input_img_3_sr)
            num_sr_imgs += 1
        else:
            print("input sr image 3 not exist!")

        if num_sr_imgs > 0:
            print("find all input sr images !")
        else:
            if os.path.exists(input_img_0):
                print("input image 0 exist!")
                image_path_list.append(input_img_0)
                num_imgs += 1
            else:
                print("input image 0 not exist!")

            if os.path.exists(input_img_1):
                print("input image 1 exist!")
                image_path_list.append(input_img_1)
                num_imgs += 1
            else:
                print("input image 1 not exist!")
            
            if os.path.exists(input_img_2):
                print("input image 2 exist!")
                image_path_list.append(input_img_2)
                num_imgs += 1
            else:
                print("input image 2 not exist!")
        
            if os.path.exists(input_img_3):
                print("input image 3 exist!")
                image_path_list.append(input_img_3)
                num_imgs += 1
            else:
                print("input image 3 not exist!")

            if num_imgs > 0:
                print("find all input images !")
            else:
                raise ValueError("input images not exist !")

        d2rgb_per_view_image_size = 416
        image_list = []
        num_idx = 0
        for image_path in image_path_list:
            image_cond, _ = process_image_path(image_path, bg_color=127, wh_ratio=0.9, use_sam=False)
            # h,w = image_cond.size
            # h = int(h*0.9)
            # w = int(h*0.9)
            # image_cond = center_crop(image_cond, w, h)
            num_idx += 1
            if args.debug_outputs or num_idx == 1:
                image_cond.save(os.path.join(output_path, f'ref_img_{num_idx}.png')) # Save before downsample
            image_cond = image_cond.resize((d2rgb_per_view_image_size, d2rgb_per_view_image_size))

            # save_seg_img_name = os.path.join(save_path_each, "seg_input_img_" + str(num_idx) + ".png")
            # image_cond.save(save_seg_img_name)

            # image_cond.save("test.png")
            # breakpoint()
            image_list.append(image_cond)
            # image_cond.save(os.path.join(output_path, f'ref_img_{num_idx}.png'))

        cond = image_list
        ref_img = image_list[0]
        # ref_img_path = image_path_list[0]
        ref_img_path = os.path.join(output_path, f'ref_img_1.png') # Use the processed ref image.


    # zero123plus w/ xyz controlnet
    d2rgb_width = d2rgb_per_view_image_size*2
    if args.three_views_input:
        d2rgb_height = d2rgb_per_view_image_size*2 # Current 3view input d2rgb model can only generate 4 views.
    else:
        d2rgb_height = d2rgb_per_view_image_size*3 # Single view input d2rgb model generate 6 views.
    image = d2rgb_pipeline(
        cond, 
        depth_image=xyz_control_image, # 
        controlnet=controlnet_d2rgb, 
        guidance_scale=3.5, # Not function if cfg_scheduler_type is enabled.
        conditioning_scale=2.0, 
        num_inference_steps=args.d2rgb_num_inference_steps, 
        width=d2rgb_width, 
        height=d2rgb_height,
        # early_stop_noise_ratio=0.1,
        generator=generator,
    ).images[0]

    # Save
    gen_output_path = os.path.join(output_path, 'final_images_d2rgb.png')

    # Copy mesh to output folder
    # os.system(f"mkdir -p {os.path.join(output_path,'final_mesh_d2rgb')}")
    # os.system(f"cp {mesh} {os.path.join(output_path,'final_mesh','final.obj')}")
    # os.system(f"cp {mesh[:-3]+'*'} {os.path.join(output_path,'final_mesh_d2rgb')+'/'}")
    # os.system(f"cp {os.path.join(output_path,f'{args.d2rgb_num_inference_steps-1}_texture.png')} {os.path.join(output_path,'final_mesh_d2rgb',f'obj_mesh_mesh_wsb_d.png')}")

    if args.debug_outputs:
        image.save(gen_output_path)
    print(gen_output_path)

    #################################### RUN PBR ####################################
    # TODO: Remove unnecessary code and test the model inference. Save albedo image.
    # breakpoint()
    pbr_img_size = 512
    n_row = 3
    n_col = 2

    img_list = []
    # img_name = os.path.join(folder_each, "res.png")
    # image = Image.open(img_name)
    # image_input = image
    # image = image.resize((img_size*2, img_size*3))
    d2rgb_image_np = np.array(image)
    d2rgb_img_size = int(d2rgb_image_np.shape[1]/2)

    for m in range(n_row):
        for n in range(n_col):
            sub_img = d2rgb_image_np[m*d2rgb_img_size: (m+1)*d2rgb_img_size, n*d2rgb_img_size: (n+1)*d2rgb_img_size, :]
            sub_img = torch.from_numpy(sub_img)
            sub_img = sub_img.permute(2, 0, 1)
            sub_img = sub_img.unsqueeze(0)
            img_list.append(sub_img)

    pbr_input_images = torch.cat(img_list, dim=0)
    pbr_input_images = pbr_input_images.unsqueeze(0)

    # Albedo
    albedo_imgs = albedo_pipeline(pbr_input_images, num_inference_steps=30, guidance_scale=0.5, width=pbr_img_size*2, height=pbr_img_size*3).images[0]
    if args.debug_outputs:
        albedo_output_path = os.path.join(output_path, 'final_images_albedo.png')
        albedo_imgs.save(albedo_output_path)
    image = albedo_imgs

    # Matallic and Roughness
    seed=20
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    metallic_roughness_imgs = metallic_roughness_pipeline(
        pbr_input_images, 
        num_inference_steps=30, 
        guidance_scale=0.5, 
        width=pbr_img_size*2, 
        height=pbr_img_size*3
    ).images[0]

    metallic_imgs, roughness_imgs, z = metallic_roughness_imgs.split()
    if args.debug_outputs:
        metallic_output_path = os.path.join(output_path, 'final_images_matallic.png')
        metallic_imgs.save(metallic_output_path)
        roughness_output_path = os.path.join(output_path, 'final_images_roughness.png')
        roughness_imgs.save(roughness_output_path)

    # Enhance metallic and roughness images
    metallic_enhancer = ImageEnhance.Contrast(metallic_imgs)
    metallic_imgs = metallic_enhancer.enhance(2.0)
    roughness_enhancer = ImageEnhance.Contrast(roughness_imgs)
    roughness_imgs = roughness_enhancer.enhance(2.0)

    # Bake matallic and roughness
    metallic_roughness_z_grid = torch.tensor(np.array(metallic_roughness_imgs)).to('cuda').permute(2,0,1).unsqueeze(0).float()/255.0
    metallic_roughness_z_texture, metallic_roughness_z_rendered = d2rgb_pipeline.scheduler.bake_pbr_mr(metallic_roughness_z_grid)
    # Save metallic and roughness texture map.
    metallic_texture_map = Image.fromarray((metallic_roughness_z_texture[0,:,:,0].cpu().numpy()*255).astype(np.uint8))
    roughness_texture_map = Image.fromarray((metallic_roughness_z_texture[0,:,:,1].cpu().numpy()*255).astype(np.uint8))
    metallic_texture_output_path = os.path.join(output_path, 'metallic_texture_map.png')
    roughness_texture_output_path = os.path.join(output_path, 'roughness_texture_map.png')
    metallic_texture_map.save(metallic_texture_output_path)
    roughness_texture_map.save(roughness_texture_output_path)

    #################################### RUN SDXL ####################################

    ref_img_file = ref_img_path
    if prompt_file is not None:
        prompt = open(prompt_file, 'r', encoding='utf-8').read()
    else:
        # prompt = ''
        # Use ChatGPT to generate prompt
        prompt = gpt_caption(ref_img_path, retries=4, default="")
        print("\n[GPT] Prompt:",prompt,"\n")

        # Replace ref_img with the gen3d processed one, if exists.
        gen3d_img_file = os.path.join(os.path.dirname(ref_img_file), 'gen3d', '0.png')
        if os.path.exists(gen3d_img_file):
            print("[SDXL] Reference image replaced by ", gen3d_img_file)

            ref_img = Image.open(gen3d_img_file).convert("RGB")
            if args.debug_outputs:
                ref_img.save(os.path.join(output_path, 'ref_img_sdxl_replaced.png'))
        else:
            print("[SDXL] Reference image not replaced", gen3d_img_file)

    # Run ref image albedo pipeline
    # Temporarily disabled because the model does not perform very well.
    # ref_image_tensor = np.array(ref_img)
    # ref_image_tensor = torch.from_numpy(ref_image_tensor)
    # ref_image_tensor = ref_image_tensor.permute(2, 0, 1)
    # ref_image_tensor = ref_image_tensor.unsqueeze(0)
    # ref_image_tensor = F.interpolate(ref_image_tensor, size=(1280, 1280), mode='nearest')
    # ref_albedo_img = ref_albedo_pipeline(ref_image_tensor, num_inference_steps=30, guidance_scale=0.5, width=1280, height=1280).images[0]
    # ref_albedo_img.save(os.path.join(output_path, 'ref_img_albedo.png'))
    # ref_img = ref_albedo_img

    mesh = obj_path
    print(f"Prompt: {prompt}, Mesh: {mesh}")
    # Initialize renderer in consistent scheduler
    obj_path = mesh
    render_resolution = int(args.resolution*args.renderer_extra_scale_up_factor/2)
    print("Render resolution", render_resolution)
    sdxl_per_view_image_size = int(args.resolution/2)
    print("SDXL per view resolution", sdxl_per_view_image_size)
    scale_factor = 0.9 # For d2rgb
    # sdxl_pipeline.scheduler.init_renderer(
    #     mesh=obj_path, 
    #     render_size=render_resolution, 
    #     scale_factor=scale_factor, 
    #     renderer_extra_scale_up_factor=args.renderer_extra_scale_up_factor, 
    #     d2rgb_special_scale=False, 
    #     texture_resolution=args.sdxl_texture_map_resolution,
    #     upsample_final_views=False,
    #     rotate_input_mesh=args.rotate_input_mesh,
    #     ctx = global_ctx,
    #     unwrap_uv=True
    # )
    sdxl_pipeline.scheduler.reuse_renderer(
        prev_renderer=d2rgb_pipeline.scheduler.renderer,
        mesh=obj_path, 
        render_size=render_resolution, 
        scale_factor=scale_factor, 
        renderer_extra_scale_up_factor=args.renderer_extra_scale_up_factor, 
        d2rgb_special_scale=False, 
        texture_resolution=args.sdxl_texture_map_resolution,
        upsample_final_views=False,
        rotate_input_mesh=args.rotate_input_mesh,
        ctx = global_ctx,
        unwrap_uv=False
    )
    print("[NVDIFFRAST] Using global ctx:", global_ctx)
    sdxl_pipeline.scheduler.disable_consistency = False
    sdxl_pipeline.scheduler.apply_consistent_interval = args.sdxl_apply_consistent_interval

    # Get xyz map from renderer, it's already computed when initializing the renderer
    xyz_maps = sdxl_pipeline.scheduler.xyz
    xyz_maps = F.interpolate(xyz_maps.permute(0,3,1,2), size=(sdxl_per_view_image_size, sdxl_per_view_image_size), mode='nearest').permute(0,2,3,1).contiguous()

    # xyz_maps_grid = torch.zeros([sdxl_per_view_image_size*2,sdxl_per_view_image_size*2,3],dtype=torch.float32,device='cuda')
    # xyz_maps_grid[:sdxl_per_view_image_size,:sdxl_per_view_image_size] = xyz_maps[0] # upper left
    # xyz_maps_grid[:sdxl_per_view_image_size,sdxl_per_view_image_size:] = xyz_maps[1] # upper right
    # xyz_maps_grid[sdxl_per_view_image_size:,:sdxl_per_view_image_size] = xyz_maps[2] # bottom left
    # xyz_maps_grid[sdxl_per_view_image_size:,sdxl_per_view_image_size:] = xyz_maps[3] # bottom right

    xyz_maps_grid = torch.zeros([sdxl_per_view_image_size*3,sdxl_per_view_image_size*2,3],dtype=torch.float32,device='cuda')
    xyz_maps_grid[:sdxl_per_view_image_size,:sdxl_per_view_image_size] = xyz_maps[0] # 1
    xyz_maps_grid[:sdxl_per_view_image_size,sdxl_per_view_image_size:] = xyz_maps[1] #    2
    xyz_maps_grid[sdxl_per_view_image_size:2*sdxl_per_view_image_size,:sdxl_per_view_image_size] = xyz_maps[2] # 3
    xyz_maps_grid[sdxl_per_view_image_size:2*sdxl_per_view_image_size:,sdxl_per_view_image_size:] = xyz_maps[3] #    4
    xyz_maps_grid[2*sdxl_per_view_image_size:,:sdxl_per_view_image_size] = xyz_maps[4] # 5
    xyz_maps_grid[2*sdxl_per_view_image_size:,sdxl_per_view_image_size:] = xyz_maps[5] #    6

    # Swap axis
    xyz_maps_grid = torch.stack([xyz_maps_grid[:,:,0],-xyz_maps_grid[:,:,2],xyz_maps_grid[:,:,1]],dim=-1)
    # plt.imsave("debug.png",(xyz_maps_grid.data.cpu().numpy()+1)/2)
    # 127
    background_mask = (xyz_maps_grid[:,:,0] == 0)
    # xyz_maps_grid[background_mask] = 0
    # xyz_control_image = Image.fromarray(((xyz_maps_grid.data.cpu().numpy()+1)/2*255).astype(np.uint8)).resize((args.resolution, args.resolution), Image.NEAREST)
    xyz_control_image = Image.fromarray(((xyz_maps_grid.data.cpu().numpy()+1)/2*255).astype(np.uint8))
    if args.debug_outputs:
        xyz_control_image.save(os.path.join(output_path, 'control_image_xyz.png'))

    # Get depth map from renderer, it's already computed when initializing the renderer
    depth_maps = sdxl_pipeline.scheduler.depth
    depth_maps = F.interpolate(depth_maps.permute(0,3,1,2), size=(sdxl_per_view_image_size, sdxl_per_view_image_size), mode='nearest').permute(0,2,3,1).contiguous()

    # depth_maps_grid = torch.zeros([sdxl_per_view_image_size*2,sdxl_per_view_image_size*2,1],dtype=torch.float32,device='cuda')
    # depth_maps_grid[:sdxl_per_view_image_size,:sdxl_per_view_image_size] = depth_maps[0] # upper left
    # depth_maps_grid[:sdxl_per_view_image_size,sdxl_per_view_image_size:] = depth_maps[1] # upper right
    # depth_maps_grid[sdxl_per_view_image_size:,:sdxl_per_view_image_size] = depth_maps[2] # bottom left
    # depth_maps_grid[sdxl_per_view_image_size:,sdxl_per_view_image_size:] = depth_maps[3] # bottom right

    # 6 views
    depth_maps_grid = torch.zeros([sdxl_per_view_image_size*3,sdxl_per_view_image_size*2,1],dtype=torch.float32,device='cuda')
    depth_maps_grid[:sdxl_per_view_image_size,:sdxl_per_view_image_size] = depth_maps[0] # 1
    depth_maps_grid[:sdxl_per_view_image_size,sdxl_per_view_image_size:] = depth_maps[1] #    2
    depth_maps_grid[sdxl_per_view_image_size:2*sdxl_per_view_image_size,:sdxl_per_view_image_size] = depth_maps[2] # 3
    depth_maps_grid[sdxl_per_view_image_size:2*sdxl_per_view_image_size:,sdxl_per_view_image_size:] = depth_maps[3] #    4
    depth_maps_grid[2*sdxl_per_view_image_size:,:sdxl_per_view_image_size] = depth_maps[4] # 5
    depth_maps_grid[2*sdxl_per_view_image_size:,sdxl_per_view_image_size:] = depth_maps[5] #    6

    # Normalize depth map to prevent grid lines leaking into image
    # depth_maps_grid = (depth_maps_grid - depth_maps_grid.min()) / (depth_maps_grid.max() - depth_maps_grid.min())
    depth_drange=0.1
    depth_maps_grid[depth_maps_grid>0] = depth_maps_grid[depth_maps_grid>0]*depth_drange + (255*(1-depth_drange))/2 # Center to 70% drange 
    # torch.save(depth_maps_grid, os.path.join(output_path, 'depth_maps_grid.pt'))
    # for ii in range(10): print("[DEBUG]")
    # print("Depth map min:",depth_maps_grid.min())
    # print("Depth map max:",depth_maps_grid.max())

    depth_control_image = Image.fromarray((depth_maps_grid.repeat(1,1,3).data.cpu().numpy()).astype(np.uint8)).resize((args.resolution,int(args.resolution*1.5)), Image.NEAREST)
    if args.debug_outputs:
        depth_control_image.save(os.path.join(output_path, 'control_image_depth.png')) 

    # Specify paths for consistent scheduler
    sdxl_pipeline.scheduler.mesh = mesh # Path to obj file
    sdxl_pipeline.scheduler.output_path = output_path

    # Add additional prompt
    # prompt = "four views 2x2 grid image. clear, High Resolution, 8K, 3D Render Style, 3DRenderAF, " + prompt + " Photorealistic photo with high contrast and lots of detail. Facing left, facing right, side view, left view, right view. in empty background. best clearity. masterpiece, highly detailed, complex pattern, highly detailed, highly detailed, highly detailed, complex, vivid, colorful, Blender, physically based rendering, PVC, Figma, Chibi, 3D, diffuse, 8K, high contrast, sharp, sharp, high sharpness, maximum sharpness"
    prompt = f"({prompt[:120]})++ four views 2x2 grid image. clear, High Resolution, 8K, 3D Render Style, 3DRenderAF, matte surface, matte, PVC matte surface, smooth lightning, diffuse, uniform lightning, smooth shape, fluent lines, Photorealistic photo with high contrast and lots of detail. Facing left, facing right, side view, left view, right view. in empty background. best clearity. masterpiece, highly detailed, complex pattern, highly detailed, highly detailed, highly detailed, complex, vivid, colorful, Blender, physically based rendering, PVC, Figma, Chibi, 3D, diffuse, 8K, high contrast, sharp, sharp, high sharpness, maximum sharpness"
    # prompt = "four views 2x2 grid image. Facing left, facing right, side view, left view, right view, ear." + prompt
    print("[SDXL] Prompt:",prompt)

    # Prompt weighting
    # conditioning, pooled = compel(prompt)
    # negative_conditioning, negative_pooled = compel(negative_prompt)
    # [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
    # [pooled, negative_pooled] = compel.pad_conditioning_tensors_to_same_length([pooled, negative_pooled])
    # pooled = pooled.squeeze(1)
    # negative_pooled = negative_pooled.squeeze(1)

    # if args.three_views_input == False:
    #     # Upsample using nearest
    #     enhancer = ImageEnhance.Sharpness(image)
    #     image = enhancer.enhance(args.d2rgb_image_sharpen_factor) 
    #     init_img = image.crop([0,0,d2rgb_per_view_image_size*2,d2rgb_per_view_image_size*2]).resize([args.resolution, args.resolution],resample=Image.NEAREST)
    #     init_img.save(os.path.join(output_path, 'final_images_d2rgb_upsampled_resized.png'))
    # else:

    # Upsample d2rgb image using Real-ESRGAN before feeding into SDXL.
    print("[REAL-ESRGAN] Upsampling d2rgb image...")
    # init_img = np.array(image.crop([0,0,d2rgb_per_view_image_size*2,d2rgb_per_view_image_size*2]))
    init_img = np.array(image) # Use full 6 views
    init_img = upsampler.enhance(init_img, outscale=2)[0]
    init_img = Image.fromarray(init_img)
    # init_img.save(os.path.join(output_path, 'final_images_d2rgb_upsampled.png'))
    init_img = init_img.resize([args.resolution, int(args.resolution*1.5)],resample=Image.NEAREST)
    if args.debug_outputs:
        init_img.save(os.path.join(output_path, 'final_images_d2rgb_upsampled_resized.png'))

    # Run ipadapter
    with torch.autocast(device_type="cuda"):
        image = ip_model.generate(
            # pil_image=ref_img.resize((512,512)),
            pil_image=ref_img.resize((768,768)),
            # pil_image=ref_img.resize((1024,1024)),
            prompt=prompt,
            negative_prompt='specular, highlight, spotlight, reflection, highlight, glare, high gloss, glossy, high-gloss, low quality, blurry, out of focus, low contrast, low sharpness, low resolution, nude, nsfw',
            image=init_img, 
            strength=args.sdxl_denoising_strength, # Denoising strength
            scale=args.sdxl_ip_adapter_scale, # IP-Adapter scale
            # ip_adapter_image=ref_img,
            control_image=[xyz_control_image,init_img,depth_control_image],
            controlnet_conditioning_scale=[args.sdxl_controlnet_scale_xyz,args.sdxl_controlnet_scale_tile,args.sdxl_controlnet_scale_depth],
            num_inference_steps=args.num_inference_steps,
            height=int(args.resolution*1.5),
            width=args.resolution,
            guidance_scale=24, # Not working, using cfg_scheduler_type instead.
            num_samples=1,
        )[0]

    torch.cuda.empty_cache()

    # Save
    gen_output_path = os.path.join(output_path, 'final_images_sdxl_1st_stage.png')
    if args.debug_outputs:
        image.save(gen_output_path)

    # Run second stage refine
    print("[SDXL 2nd stage refine] Start second stage refine...")
    # Reset lora
    # sdxl_pipeline.set_adapters(args.sdxl_second_refine_lora_stack, adapter_weights=args.sdxl_second_refine_lora_stack_weights)
    # Initialize renderer in consistent scheduler
    obj_path = mesh
    render_resolution = int(args.sdxl_second_refine_resolution*args.renderer_extra_scale_up_factor/2)
    print("Render resolution", render_resolution)
    sdxl_per_view_image_size = int(args.sdxl_second_refine_resolution/2)
    print("SDXL per view resolution", sdxl_per_view_image_size)
    scale_factor = 0.9 # For d2rgb
    # sdxl_pipeline.scheduler.init_renderer(
    #     mesh=obj_path, 
    #     render_size=render_resolution, 
    #     scale_factor=scale_factor, 
    #     renderer_extra_scale_up_factor=args.renderer_extra_scale_up_factor, 
    #     d2rgb_special_scale=False, 
    #     texture_resolution=args.sdxl_texture_map_resolution,
    #     upsample_final_views=True,
    #     rotate_input_mesh=args.rotate_input_mesh,
    #     ctx=global_ctx,
    #     unwrap_uv=True
    # )
    sdxl_pipeline.scheduler.reuse_renderer(
        prev_renderer=sdxl_pipeline.scheduler.renderer,
        mesh=obj_path, 
        render_size=render_resolution, 
        scale_factor=scale_factor, 
        renderer_extra_scale_up_factor=args.renderer_extra_scale_up_factor, 
        d2rgb_special_scale=False, 
        texture_resolution=args.sdxl_texture_map_resolution,
        upsample_final_views=True,
        rotate_input_mesh=args.rotate_input_mesh,
        ctx=global_ctx,
        unwrap_uv=False
    )
    print("[NVDIFFRAST] Using global ctx:", global_ctx)
    sdxl_pipeline.scheduler.disable_consistency = args.sdxl_second_refine_disable_consistency
    sdxl_pipeline.scheduler.apply_consistent_interval = args.sdxl_second_refine_apply_consistent_interval

    # Process image
    # Sharpen 
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(args.sdxl_1st_stage_output_sharpen_factor) 
    # enhancer = ImageEnhance.Contrast(image)
    # image = enhancer.enhance(1.05)
    image.resize((args.sdxl_second_refine_resolution,int(args.sdxl_second_refine_resolution*1.5)),resample=Image.NEAREST)
    gen_output_path = os.path.join(output_path, 'final_images_sdxl_1st_stage_sharpened_resized.png')
    if args.debug_outputs:
        image.save(gen_output_path)

    with torch.autocast(device_type="cuda"):
        image = ip_model.generate(
            pil_image=ref_img.resize((1024,1024)),
            prompt=prompt,
            negative_prompt='low quality, blurry, out of focus, low contrast, low sharpness, low resolution',
            image=image, 
            strength=args.sdxl_second_refine_denoising_strength, # Denoising strength
            scale=args.sdxl_second_refine_ip_adapter_scale, # IP-Adapter scale
            # ip_adapter_image=ref_img,
            control_image=[xyz_control_image,init_img,depth_control_image],
            controlnet_conditioning_scale=[args.sdxl_second_refine_controlnet_scale_xyz,args.sdxl_second_refine_controlnet_scale_tile,args.sdxl_second_refine_controlnet_scale_depth],
            num_inference_steps=args.sdxl_second_refine_num_inference_steps,
            height=int(args.sdxl_second_refine_resolution*1.5),
            width=args.sdxl_second_refine_resolution,
            guidance_scale=24, # Not working, using cfg_scheduler_type instead.
            num_samples=1,
        )[0]
 
    # Save
    gen_output_path = os.path.join(output_path, 'final_images.png')

    # Copy mesh to output folder
    # os.system(f"mkdir -p {os.path.join(output_path,'final_mesh')}")
    # os.system(f"cp {mesh} {os.path.join(output_path,'final_mesh','final.obj')}")
    # os.system(f"cp {mesh[:-3]+'*'} {os.path.join(output_path,'final_mesh')+'/'}")
    # os.system(f"cp {os.path.join(output_path,f'{args.sdxl_second_refine_num_inference_steps-1}_texture.png')} {os.path.join(output_path,'final_mesh',f'kd.png')}")

    image.save(gen_output_path)
    print(gen_output_path)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"[TIME] running time: {elapsed_time} seconds")     

    # # [NEW] Save PBR mesh (Deprecated, using overwrite_mtl after copy to gdp)
    # rgb_texture = sdxl_pipeline.scheduler.last_texture_map
    # mrz_texture = F.interpolate(metallic_roughness_z_texture.permute(0,3,1,2), size=(rgb_texture.shape[0], rgb_texture.shape[1]), mode='bicubic')[0].permute(1,2,0)
    # pbr_texture = torch.cat([rgb_texture, mrz_texture[:,:,1:2],mrz_texture[:,:,0:1]], dim=-1)
    # mesh_export_path = os.path.join(output_path, 'pbr_mesh', f'{job_id}.obj')
    # os.makedirs(os.path.join(output_path, 'pbr_mesh'),exist_ok=True)
    # sdxl_pipeline.scheduler.renderer.export_mesh(mesh_export_path,pbr_texture)
    # # sdxl_pipeline.scheduler.renderer.overwrite_mtl(mesh_export_path, pbr_texture, val_range=(0,1), texture_prefix="texture", default_material_name="raw_mesh")

    if input_has_uv:
        # Copy to gdp
        print("Copying to gdp...")
        print(f"{out_dir}/{job_id}.obj")
        os.system(f"cp {mesh[:-3]+'obj'} {out_dir}/{job_id}.obj")
        os.system(f"cp {mesh[:-3]+'mtl'} {out_dir}/")
        # Find the copied mtl file
        mtl_file = mesh[:-3] + 'mtl'  # Extract the mtl filename from the mesh file
        copied_mtl_path = os.path.join(out_dir, os.path.basename(mtl_file))
        # Modify the mtf file by removeing some lines and add some new lines
        with open(mtl_file, 'r') as file:
            lines = file.readlines()
        with open(copied_mtl_path, 'w') as file:
            for line in lines:
                file.write(line) # Now we copy mtl to new location without modifying it.
                                # It will be modified by following overwrite_mtl function.

                # if "map_d" in line:
                #     file.write("map_d texture.png\n")
                # elif "map_Kd" in line:
                #     file.write("map_Kd texture.png\n")
                # elif "Ns" in line: # Remove Ns
                #     pass
                # elif "Ka" in line: # Remove Ka
                #     pass
                # elif "Kd" in line: # Remove Kd
                #     pass
                # elif "Ks" in line: 
                #     file.write("Ks 0 0 0\n")
                # elif "Ke" in line: # Remove Ke
                #     pass
                # elif "Ni" in line: # Remove Ni
                #     pass
                # elif "d " in line: # Remove d
                #     pass
                # elif "illum" in line: # Remove illum
                #     pass
                # else:
                #     file.write(line)

        # tdmq_out_texmap = os.path.join(out_dir, "texture.png")
        # os.system(f"cp {os.path.join(output_path,f'{args.sdxl_second_refine_num_inference_steps-1}_texture.png')} {tdmq_out_texmap}")
        
        # [NEW] Overwrite mtl and save pbr texture
        rgb_texture = sdxl_pipeline.scheduler.last_texture_map
        mrz_texture = F.interpolate(metallic_roughness_z_texture.permute(0,3,1,2), size=(rgb_texture.shape[0], rgb_texture.shape[1]), mode='bicubic')[0].permute(1,2,0)
        pbr_texture = torch.cat([rgb_texture, mrz_texture[:,:,1:2],mrz_texture[:,:,0:1]], dim=-1)
        texture_prefix = os.path.join(out_dir, "texture")
        sdxl_pipeline.scheduler.renderer.overwrite_mtl(copied_mtl_path, pbr_texture, val_range=(0,1), texture_prefix=texture_prefix, default_material_name="raw_mesh")
    
    else:
        # [NEW] Export mesh
        mesh_export_path = f"{out_dir}/{job_id}.obj"
        rgb_texture = sdxl_pipeline.scheduler.last_texture_map
        mrz_texture = F.interpolate(metallic_roughness_z_texture.permute(0,3,1,2), size=(rgb_texture.shape[0], rgb_texture.shape[1]), mode='bicubic')[0].permute(1,2,0)
        pbr_texture = torch.cat([rgb_texture, mrz_texture[:,:,1:2],mrz_texture[:,:,0:1]], dim=-1)
        sdxl_pipeline.scheduler.renderer.export_mesh(mesh_export_path, pbr_texture)

    # Debug: link output folder to local output folder
    os.system(f"ln -s {os.path.abspath(os.path.dirname(out_dir))} {output_path}/gdp_out_dir")
    print("Done. ")

    # Clean up
    print("[FINISH] Cleaning up...")
    # breakpoint()
    # Delete RasterizeCudaContext
    # print("[FINISH] Reference count of global_ctx:", sys.getrefcount(global_ctx))
    # print(gc.get_referrers(global_ctx))
    # import referrers
    # referrers.get_referrer_graph(global_ctx)
    # Delete every renderer
    # del d2rgb_pipeline.scheduler.renderer.ctx
    del sdxl_pipeline.scheduler.renderer.ctx
    # del d2rgb_pipeline.scheduler.renderer
    del sdxl_pipeline.scheduler.renderer
    del global_ctx
    gc.collect()
    torch.cuda.empty_cache()
    print("[FINISH] Clean up done.")


if __name__ == '__main__':


    # Parse args
    parser = argparse.ArgumentParser()
    # 3view sample
    # parser.add_argument('--input_link', type=str, default='', help='Path to input obj folder') 
    # 1view img_to_3D sample
    
    parser.add_argument('--obj_path', type=str, default='', help='Path to input obj') 
    parser.add_argument('--ref_img_path', type=str, default='', help='Path to input reference image') 
    parser.add_argument('--output_path', type=str, default='', help='Path to output folder') 
    parser.add_argument('--repeat', type=int, default=1, help='Repeat times')

    args = parser.parse_args()

    models = init_models(three_views_input=False, debug_outputs=True)

    # Print cuda memory usage
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    mem_used_mb = (total - free) / 1024 ** 2
    print("[CUDA] Init. Memory used: ",mem_used_mb)
    # torch.cuda.memory_summary()

    for repeat_time in range(args.repeat):
        print("Start inference for repeat ",repeat_time)
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        current_time+="-["+str(repeat_time)+"]"
        output_path = os.path.join("./outputs_service_d2rgb_6views_sdxl_sr","debug_service_out",current_time,"texture_mesh")
        os.makedirs(output_path, exist_ok=True)

        run_d2rgb_sdxl_sr(
            models,
            args.obj_path,
            args.ref_img_path,  
            args.output_path,
            job_id=f'job_{current_time}',
            seed=0
        )

        # Print cuda memory usage
        device = torch.device('cuda:0')
        free, total = torch.cuda.mem_get_info(device)
        mem_used_mb = (total - free) / 1024 ** 2
        print(f"[CUDA] Finished. Memory used: ",mem_used_mb)
        # torch.cuda.memory_summary()
