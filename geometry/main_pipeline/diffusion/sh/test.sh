
## test multi view cond diffusion
# CUDA_VISIBLE_DEVICES=0 \
# python scripts/test_mmdit_image23D_flow_4views.py \
#     --exp_dir "configs/4view_gray_2048_flow" \
#     --save_dir "configs/4view_gray_2048_flow" \
#     --image_dir "/data/validation/images_mv"



# ## test 1 view cond diffusion
CUDA_VISIBLE_DEVICES=0 \
python scripts/test_mmdit_image23D_flow_1view.py \
    --exp_dir "../../../../Tencent_XR_3DGen/geometry_dit" \
    --save_dir "./outputs/" \
    --image_dir "./sample_images/"
