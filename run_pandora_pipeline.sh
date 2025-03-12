# Fix functional_tensors bug
PYTHON_PATH=$(which python)
SITE_PACKAGES_PATH=$($PYTHON_PATH -c "import site; print(site.getsitepackages()[0])")
TARGET_FILE="$SITE_PACKAGES_PATH/basicsr/data/degradations.py"

if [ -f "$TARGET_FILE" ]; then
    sed -i 's/functional_tensor/functional/g' "$TARGET_FILE"
    echo "Modification completed in $TARGET_FILE"
else
    echo "File not found: $TARGET_FILE"
fi

# Shape Generation
cd geometry/main_pipeline/diffusion
sh sh/test.sh
cd -

# Texture Generation
cd texture/tex_refine
python inference_consistent_d2rgb_6views_sdxl_sr_v5_pbr.py \
--obj_path '../../geometry/main_pipeline/diffusion/outputs/test_out_2025-03-11-05:01:40_checkpoint-1/test_images/typical_creature_robot_crab/mesh.obj' \
--ref_img_path '../../geometry/main_pipeline/diffusion/outputs/test_out_2025-03-11-05:01:40_checkpoint-1/test_images/typical_creature_robot_crab/typical_creature_robot_crab.png' \
--output_path './outputs'
cd -

