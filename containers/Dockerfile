###############################################################################
# A first image to build the planner
###############################################################################
FROM ubuntu:20.04 AS builder

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"
ENV CXX g++

WORKDIR /workspace/

RUN apt-get update && apt-get install --no-install-recommends -y \
	build-essential \
	cmake \
	ca-certificates \
	curl \
	python3 python3-pip python3-dev python3-setuptools \
	scons \
	git \
#	openssh-client \
	pkg-config \
	libboost-program-options-dev libboost-filesystem-dev libboost-system-dev \
	libboost-chrono-dev libboost-timer-dev libboost-serialization-dev \
    # OpenWbo:
    libz-dev libgmp-dev \
    && rm -rf /var/lib/apt/lists/*


# Download & build OpenWbo
#################################
RUN curl -SL https://github.com/sat-group/open-wbo/archive/master.tar.gz | tar -xz \
	&& mv open-wbo-master open-wbo \
	&& cd open-wbo \
	&& make rs

# Install & build the FS planner
#################################
# Add previously-cloned FS directory
RUN git clone --depth 1 -b sltp-lite --single-branch https://github.com/aig-upf/fs-private.git fs \
    && cd fs \
    && git submodule update --init \
    && python3 ./build.py -p

# Download and Install D2L
#################################
# We install Python dependencies in user space so that we can move them to the "runner" image more easily
# (see https://pythonspeed.com/articles/multi-stage-docker-python/)
RUN git clone --depth 1 -b main --single-branch https://github.com/rleap-project/d2l \
    && cd d2l \
    && pip3 install --user -e .



###############################################################################
# The final image to run the planner
###############################################################################
FROM ubuntu:20.04 AS runner
MAINTAINER Guillem Francès guillem.frances@upf.edu

WORKDIR /workspace/

RUN apt-get update && apt-get install --no-install-recommends -y \
	python3 \
	libboost-program-options-dev libboost-filesystem-dev libboost-system-dev \
	libboost-chrono-dev libboost-timer-dev libboost-serialization-dev \
	# If we transition to the latest FS code, only these will be necessary:
    # libboost-program-options1.71.0 \
    # libboost-filesystem1.71.0 \
    # libboost-timer1.71.0 \
    vim \
    && rm -rf /var/lib/apt/lists/*


# Copy the relevant files from the previous docker build into this build.
COPY --from=builder /workspace/open-wbo/open-wbo_static /bin/
COPY --from=builder /workspace/fs/ /workspace/fs/
COPY --from=builder /workspace/d2l/ /workspace/d2l/

# Copy all Python packages that were locally installed on the "builder" Docker image:
COPY --from=builder /root/.local /root/.local
# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH

# Expose the FS planner for the rest of components of the container
ENV FS_PATH="/workspace/fs/"

#COPY ./docker-entrypoint.sh /
#ENTRYPOINT ["/docker-entrypoint.sh"]
ENTRYPOINT ["python3","/workspace/d2l/experiments/run.py"]