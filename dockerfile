ARG image
FROM $image

ARG DEBIAN_FRONTEND=noninteractive

ARG gpu_arch
ARG cores
RUN apt update -y

# compile opencv

RUN apt install -y build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-dev python3-pip python3-tk

WORKDIR /opencv
RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git

WORKDIR /opencv/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF  \
-D OPENCV_ENABLE_NONFREE=ON  \
-D WITH_CUDA=ON  \
-D WITH_CUDNN=ON  \
-D OPENCV_DNN_CUDA=ON  \
-D ENABLE_FAST_MATH=1  \
-D CUDA_FAST_MATH=1  \
-D CUDA_ARCH_BIN=$gpu_arch \
-D WITH_CUBLAS=1  \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules  \
-D HAVE_opencv_python3=ON  \
-D BUILD_EXAMPLES=ON ..

RUN make j$cores
RUN make install

WORKDIR /app

COPY . .
CMD ["python3", "main.py"]