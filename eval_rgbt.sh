# test lasher
CUDA_VISIBLE_DEVICES=0,1 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name tgatrack --dataset_name LasHeR --yaml_name rgbt

# test rgbt234
CUDA_VISIBLE_DEVICES=0,1 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name tgatrack --dataset_name RGBT234 --yaml_name rgbt