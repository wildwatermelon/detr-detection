# Project notes

### update requirement.txt
`pip3 freeze > requirements.txt`<br>
`pip install -r requirements.txt` <br>
`pycocotools @ git+https://github.com/waleedka/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI` <br>

### docker cmd
docker build -f Dockerfile -t detr-detection:latest .
docker run --gpus all -itd [docker_image]

#### docker bind mount
docker run --gpus all --shm-size 8G -itd -v D:/ITSS/balloon_dataset:/workspace/detr-detection/datasets detr-detection

docker exec -it [docker_image] bash
cd mnt/c/users/kjiak/nus-iss/Project2/detr-detection/

### docker copy
docker cp container_id:/foo.txt foo.txt

### dump into a Bash session instead of inside Python 3.9 REPL
CMD ["/bin/bash"] replace with docker exec

### pycoco docker git solution
`pip install matplotlib`
`https://github.com/matterport/Mask_RCNN/issues/6`
`On Linux, run pip install git+https://github.com/waleedka/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI`
`On Windows, run pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI`
git clone https://token@github.com/pdollar/coco.git <br>
cd coco/PythonAPI <br>
python setup.py install <br>

### pip
`pip install matplotlib requests scipy pycocotools torchvision torchaudio`

### tensorboard
`python -m tensorboard.main --logdir='D:\ITSS\detr-detection\lightning_logs\version_0'`

### remove segmentation
remove segmentation from coco json

### nvidia-smi (check gpu)
nvidia-smi
