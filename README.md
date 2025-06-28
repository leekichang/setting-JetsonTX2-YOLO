# setting-JetsonTX2-YOLO

setting Jetson TX2 for YOLO test

# JetPack 4.x용 YOLO 컨테이너 실행

docker run --rm -it --runtime nvidia --ipc=host -v $(pwd):/workspace ultralytics/ultralytics:latest-jetson-jetpack4

yolo predict model=yolov8n.pt source=0



Python 3.8 가상환경 만들기

sudo apt-get install -y python3.8 python3.8-venv python3.8-dev

python3.8 -m venv ~/yoloenv

source ~/yoloenv/bin/activate

pip install -U pip wheel

Jetson 전용 PyTorch & torchvision wheel 설치

(JetPack 4.6.4 + CUDA 10.2 기준 예시)


wget https://nvidia.box.com/shared/static/v7m6sgwql3xxnwr84fh6pwbs92ne0u0f.whl -O torch-1.13.0+nv23.08-cp38-cp38-linux_aarch64.whl

pip install torch-1.13.0+nv23.08-cp38-cp38-linux_aarch64.whl

pip install torchvision==0.14.0+nv23.08 --extra-index-url https://pypi.ngc.nvidia.com

(필요한 wheel 목록은 “PyTorch for Jetson” 포럼 글에 버전별로 정리돼 있습니다.) 

forums.developer.nvidia.com

Ultralytics 설치 (Torch 1.13과 호환되는 마지막 버전)

pip install -U numpy==1.23.5          # pandas 의존성 충돌 방지

pip install ultralytics==8.1.27

동작 확인

python

복사

편집

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

print(model('https://ultralytics.com/images/bus.jpg')[0].boxes)
