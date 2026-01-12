for level in  'large';do
echo $level
auto_mask_start_time=$(date +%s)
python auto-mask-batch.py \
    --video_path videos/chickenchicken \
    --output_dir output/chickenchicken \
    --batch_size 40 \
    --detect_stride 10 \
    --level ${level}
auto_mask_end_time=$(date +%s)
auto_mask_elapsed_time=$((auto_mask_end_time - auto_mask_start_time))
echo "auto-mask-batch.py execution time: ${auto_mask_elapsed_time} seconds"

# visulization_start_time=$(date +%s)
# python visulization.py \
#     --video_path videos/chickenchicken \
#     --output_dir output/chickenchicken \
#     --level ${level}
# visulization_end_time=$(date +%s)
# visulization_elapsed_time=$((visulization_end_time - visulization_start_time))
# echo "visulization.py execution time: ${visulization_elapsed_time} seconds"
done