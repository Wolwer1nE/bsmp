FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt update && apt install cmake gdb git wget \
    build-essential autoconf libssl-dev libyaml-dev zlib1g-dev libffi-dev libgmp-dev \
    lsb-release wget software-properties-common gnupg ninja-build bash \
    python3 python3-pip python3-venv -y
    
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

RUN apt install clang-tidy-22