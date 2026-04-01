FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt update && apt install cmake gdb git wget \
    build-essential autoconf libssl-dev libyaml-dev zlib1g-dev libffi-dev libgmp-dev -y

# Ruby installation
RUN git clone https://github.com/rbenv/rbenv.git ~/.rbenv && \
    echo 'eval "$(~/.rbenv/bin/rbenv init - --no-rehash bash)"' >> ~/.bashrc && \
    git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build && \
    ~/.rbenv/bin/rbenv install 3.3.10 && \
    ~/.rbenv/bin/rbenv global 3.3.10

# Julia installation
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.5-linux-x86_64.tar.gz && \
    tar -xzf julia-1.12.5-linux-x86_64.tar.gz -C /opt/ && \
    ln -s /opt/julia-1.12.5/bin/julia /usr/local/bin/julia && \
    rm julia-1.12.5-linux-x86_64.tar.gz
