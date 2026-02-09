FROM nvidia/cuda:13.1.1-devel-ubuntu22.04
WORKDIR /app

RUN apt update
RUN apt upgrade -y

# Install pip (Python3 is included in ubuntu without pip)
RUN apt install -y python3-pip

# Helper tool for querying installed libraries.
RUN apt install -y pkg-config

# Install dependencies for blender.
RUN apt install -y \
    wget \
    xz-utils \
    libxfixes3 \
    libxi6 \
    libxkbcommon0 \
    libgl1 \
    libglx0 \
    libopengl0 \
    libsm6 \
    libice6 \
    libxrender1 \
    libgomp1 \
    libxcursor1 \
    libxinerama1 \
    libegl1

# Download blender 5.0.1 from extracted mirror
RUN wget -q https://ftp.halifax.rwth-aachen.de/blender/release/Blender5.0/blender-5.0.1-linux-x64.tar.xz -O /tmp/blender.tar.xz
RUN tar -xvf /tmp/blender.tar.xz -C /opt
RUN ln -s /opt/blender-5.0.1-linux-x64/blender /usr/local/bin/blender

# Install python libraries
RUN pip install numpy
RUN pip install pillow
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# OpenCV dependencies
RUN apt install -y cmake g++ wget unzip
RUN apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libx264-dev

# Download OpenCV code
RUN mkdir /root/opencv
RUN wget -O /root/opencv/opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
RUN wget -O /root/opencv/opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
RUN unzip /root/opencv/opencv.zip -d /root/opencv/
RUN unzip /root/opencv/opencv_contrib.zip -d /root/opencv/
RUN mkdir /root/opencv/build

# Configure OpenCV
RUN cd /root/opencv/build && cmake -DWITH_FFMPEG=1 -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x

# Build and install OpenCV
RUN cd /root/opencv/build && make -j$(nproc)
RUN cd /root/opencv/build && make install

RUN pip install matplotlib

ENTRYPOINT sleep infinity
