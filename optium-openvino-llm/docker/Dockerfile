FROM ubuntu:22.04
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=america/los_angeles

# #1 install base packages
#-----------------------
RUN apt-get update && \
    apt-get install --no-install-recommends -q -y \
    software-properties-common \
    ca-certificates \
    gnupg \
    wget \
    python3 \
    python3-pip

# #2 Install Intel GPU compute user-space drivers
#-----------------------
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
   gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
   echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
   tee /etc/apt/sources.list.d/intel-gpu-jammy.list
# apt-get update && \
#  apt-get install --no-install-recommends -q -y \
#    intel-opencl-icd intel-level-zero-gpu level-zero
# *Defect/Workaround*: Intel's apt repo does not contain the latest GPU runtime supporting kernel 6.8, so install the packages directly from their compute
# runtime repo
RUN apt-get update && \
  apt-get install --no-install-recommends -q -y \
    ocl-icd-libopencl1 \
    clinfo && \
  cd /tmp && \
  wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.16510.2/intel-igc-core_1.0.16510.2_amd64.deb && \
  wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.16510.2/intel-igc-opencl_1.0.16510.2_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/24.13.29138.7/intel-level-zero-gpu_1.3.29138.7_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/24.13.29138.7/intel-opencl-icd_24.13.29138.7_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/24.13.29138.7/libigdgmm12_22.3.18_amd64.deb && \
  dpkg -i *.deb && \
  apt-get install --no-install-recommends -q -y \
    level-zero 

# Required compute runtime level-zero variables
ENV ZES_ENABLE_SYSMAN=1

# #3 Chatbot specific dependencies
#-----------------------
RUN pip3 install --upgrade-strategy eager optimum[openvino,nncf] gradio 
