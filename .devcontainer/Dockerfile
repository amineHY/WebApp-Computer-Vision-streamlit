FROM nvcr.io/nvidia/tensorrt:19.09-py3
LABEL maintainer "M. Amine Hadj-Youcef  <hadjyoucef.amine@gmail.com>"

# If you have any comment : LinkedIn - https://www.linkedin.com/in/aminehy/

ARG DEBIAN_FRONTEND=noninteractive


ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

##---------------install prerequisites---------------

RUN apt-get -qq update && apt-get -qq install -y --no-install-recommends \
	protobuf-compiler \
	geany \
	python3 \
	python3-tk \
	python3-pip \
	python3-dev \
	python3-setuptools \
	eog \
	gedit \
	build-essential \
	ssh \
	ca-certificates \
	curl \
	cmake \
	git \
	wget \
	unzip \
	yasm \
	pkg-config \
	libswscale-dev \
	libtbb2 \
	libtbb-dev \
	libjpeg-dev \
	libgflags-dev \
	libgoogle-glog-dev \
	libprotobuf-dev \
	liblmdb-dev \
	libpng-dev \
	libtiff-dev \
	libavformat-dev \
	libpq-dev \
	libgtk2.0-dev \
	libhdf5-dev \
	libcurl4-openssl-dev\
	libprotoc-dev \
	swig\
	qt5-default \
	libboost-all-dev \
	libboost-dev \
	xdg-utils \
	snapd \
	rsync \
	&& rm -rf /var/lib/apt/lists/*

RUN cd /usr/local/bin &&\
	ln -s /usr/bin/python3 python


#---------------Install pip package---------------
RUN cd /usr/local/src \
	&& wget -q  https://bootstrap.pypa.io/get-pip.py \
	&& python3 get-pip.py \
	&& pip3 install --upgrade pip \
	&& rm -f get-pip.py \
	&& python3 -m pip --version\
	&& pip install --user --upgrade pip

RUN pip3 install --upgrade pip


#---------------Install opencv----------------------
WORKDIR /
ENV OPENCV_VERSION="4.1.1"
RUN wget -O opencv.zip  https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip 
# RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip 
RUN unzip opencv.zip 
# RUN unzip opencv_contrib.zip 
RUN mkdir /opencv-${OPENCV_VERSION}/cmake_binary 
WORKDIR /opencv-${OPENCV_VERSION}/cmake_binary 

RUN cmake -DBUILD_TIFF=ON \
	-DBUILD_opencv_java=OFF \
	-DWITH_CUDA=ON \
	-DENABLE_FAST_MATH=1 \
	-DCUDA_FAST_MATH=1 \
	-DWITH_CUBLAS=1 \
	-DENABLE_AVX=ON \
	-DWITH_OPENGL=ON \
	-DWITH_OPENCL=ON \
	-DWITH_IPP=ON \
	-DWITH_TBB=ON \
	-DWITH_EIGEN=ON \
	-DWITH_V4L=ON \
	-DBUILD_TESTS=OFF \
	-DBUILD_PERF_TESTS=OFF \
	-DCMAKE_BUILD_TYPE=RELEASE \
	-DCMAKE_INSTALL_PREFIX=$(python3.6 -c "import sys; print(sys.prefix)") \
	-DPYTHON_EXECUTABLE=$(which python3.6) \
	-DPYTHON_INCLUDE_DIR=$(python3.6 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
	-DPYTHON_PACKAGES_PATH=$(python3.6 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
	-DINSTALL_PYTHON_EXAMPLES=ON \
	-DINSTALL_C_EXAMPLES=OFF \
	-DOPENCV_ENABLE_NONFREE=ON \
	-DOPENCV_GENERATE_PKGCONFIG=ON \
	# -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib-${OPENCV_VERSION}/modules \
	-DBUILD_EXAMPLES=ON \
	-D CUDA_TOOLKIT_ROOT_DIR= /usr/local/cuda-10.1 \
	-DWITH_QT=ON ..

RUN chmod +x download_with_curl.sh \
	&& sh ./download_with_curl.sh

RUN make -j8 \
	&& make install \
	&& rm /opencv.zip \
	# && rm opencv_contrib.zip \
	&& rm -rf /opencv-${OPENCV_VERSION} 
	# && rm -rf /opencv_contrib-${OPENCV_VERSION}

RUN  ln -s \
	/usr/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-36m-x86_64-linux-gnu.so \
	/usr/local/lib/python3.6/dist-packages/cv2.so


####################################################
# Deep learning frameworks
####################################################

#---------------Install PyTorch---------------------
RUN pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

#---------------Install TensorFlow------------------
RUN pip3 install tensorflow-gpu
RUN pip3 install tflearn

#---------------Install ONNX------------------------
RUN pip3 install onnx onnxmltools onnxruntime-gpu

#---------------Install keras------------------------
RUN pip3 install keras

#---------------Install ONNX-TensorRT---------------
# determine DGPU_ARCHS from https://developer.nvidia.com/cuda-gpus
# https://github.com/onnx/onnx-tensorrt

RUN	git clone --recursive -b 6.0 https://github.com/onnx/onnx-tensorrt.git &&\
	cd onnx-tensorrt &&\
	mkdir build  &&\
	cd build &&\
	cmake .. -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include -DTENSORRT_ROOT=/usr/src/tensorrt -DGPU_ARCHS="61" &&\
	make -j8 &&\
	make install &&\
	ldconfig && \
	cd .. && \
	python setup.py build &&\
	python setup.py install &&\
	rm -rf ./build/

#----------------Install TensorBoardX -----------------------
RUN git clone https://github.com/lanpa/tensorboardX && cd tensorboardX && python setup.py install

#---------------Install python requirements---------
COPY requirements.txt /tmp/
RUN pip3 install --requirement /tmp/requirements.txt

RUN echo 'export LC_ALL=C.UTF-8' >> ~/.bashrc
RUN echo 'export LANG=C.UTF-8' >> ~/.bashrc


#----------------Perform some cleaning-----------------------
RUN (apt-get -qq autoremove -y; \
	apt-get -qq autoclean -y)

#----------------set the working directory-------------------
WORKDIR /workspace

CMD ["jupyter", "notebook", "--allow-root", "--port=8888", "--ip=0.0.0.0", "--no-browser"]
