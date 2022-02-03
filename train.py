# num_classes = 1
# !python main.py \
#   --dataset_file "custom" \
#   --coco_path "/content/data/custom/" \
#   --output_dir "outputs" \
#   --resume "detr-r50_no-class-head.pth" \
#   --num_classes $num_classes \
#   --epochs 10
#
# python ./detr/main.py --dataset_file "custom" --coco_path "D:/ITSS/balloon_dataset/balloon" --output_dir "outputs" --resume
# "detr-r50_no-class-head.pth" --num_classes 1 --epochs 10
#
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./datasets/balloon" --output_dir "outputs"
# --resume "detr-r50_no-class-head.pth" --num_classes 1 --epochs 10
#
# python3 ./detr/main.py --dataset_file "custom" --coco_path "./datasets/balloon" --output_dir "outputs100"
# --resume "detr-r50_no-class-head.pth" --num_classes 1 --epochs 100