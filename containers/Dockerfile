###############################################################################
# A first image to build the planner
###############################################################################
FROM ubuntu:20.04 AS builder

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"
RUN apt-get update && apt-get install --no-install-recommends -y \
	build-essential \
	cmake \
	ca-certificates \
	curl \
	python3 python3-pip python3-dev python3-setuptools \
	scons \
	git \
	openssh-client \
	pkg-config \
	libboost-program-options-dev libboost-filesystem-dev libboost-system-dev \
	libboost-chrono-dev libboost-timer-dev libboost-serialization-dev \
    # OpenWbo:
    vim libz-dev libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/

# Set up some environment variables.
ENV CXX g++

#################################
# Install & build OpenWbo
#################################
RUN curl -SL https://github.com/sat-group/open-wbo/archive/master.tar.gz | tar -xz \
	&& mv open-wbo-master open-wbo \
	&& cd open-wbo \
	&& make rs \
    && cp ./open-wbo_static /bin/

#################################
# Install & build the devel branch of Tarski
#################################
RUN git clone --depth 1 -b devel --single-branch https://github.com/aig-upf/tarski.git tarski

#################################
# Install & build SLTP
#################################
RUN git clone --depth 1 -b d2l --single-branch https://github.com/aig-upf/sltp.git d2l \
    && cd d2l \
    && cd src/generators \
    && cmake . \
    && make -j4 \
    && rm -rf CMakeFiles

#################################
# Install & build the FS planner
#################################
# Add previously-cloned FS directory
RUN git clone --depth 1 -b sltp-lite --single-branch https://github.com/aig-upf/fs-private.git fs \
    && cd fs \
    && git submodule update --init \
    && python3 ./build.py -p

###############################################################################
# The final image to run the planner
###############################################################################
FROM ubuntu:20.04 AS runner
MAINTAINER Guillem Francès guillem.frances@upf.edu


ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"
ENV FS_PATH="/workspace/fs/"

RUN apt-get update && apt-get install --no-install-recommends -y \
	build-essential \
	cmake \
	python3 python3-pip python3-dev python3-setuptools \
	libboost-program-options-dev libboost-filesystem-dev libboost-system-dev \
	libboost-chrono-dev libboost-timer-dev libboost-serialization-dev \
    # OpenWbo:
    vim libz-dev libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/

# Copy the relevant files from the previous docker build into this build.
COPY --from=builder /workspace/open-wbo/open-wbo_static /bin/
COPY --from=builder /workspace/tarski/ /workspace/tarski/
COPY --from=builder /workspace/d2l/ /workspace/d2l/
COPY --from=builder /workspace/fs/ /workspace/fs/

# Install the wheel module for subsequents pip installs
RUN pip3 install wheel && pip3 install -e /workspace/tarski/ && pip3 install -e /workspace/d2l/

#ENTRYPOINT ["sh", "-c", "$SLTP_PATH/experiments/run.py"]
#CMD ["bash"]
COPY ./docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
