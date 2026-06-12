FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt update && apt install cmake gdb git wget \
    build-essential autoconf libssl-dev libyaml-dev zlib1g-dev libffi-dev libgmp-dev \
    lsb-release wget software-properties-common gnupg ninja-build bash -y
    
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

RUN apt install clang-tidy-22

RUN git clone https://github.com/rbenv/rbenv.git ~/.rbenv

RUN echo 'eval "$(~/.rbenv/bin/rbenv init - --no-rehash bash)"' >> ~/.bashrc

RUN git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build

RUN ~/.rbenv/bin/rbenv install 3.3.10

RUN ~/.rbenv/bin/rbenv global 3.3.10
