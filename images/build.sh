#!/usr/bin/env bash
set -e -x


SLTP_DIR=`pwd`/..

# the temp directory that we will use
WORK_DIR=`mktemp -d`

# check if tmp dir was created
if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

# deletes the temp directory
function cleanup {
  #rm -rf "$WORK_DIR"
  echo "Deleted temp working directory $WORK_DIR"
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

pushd $WORK_DIR

git clone --depth 1 -b dev-0.2.0-symbols --single-branch git@github.com:aig-upf/tarski.git tarski
git clone --depth 1 -b sltp-lite --single-branch git@github.com:aig-upf/fs-private.git fs
cd fs && git submodule update --init && cd ..

# git clone --depth 1 -b integrating-with-tarski --single-branch git@github.com:aig-upf/sltp.git sltp
mkdir sltp
git -C $SLTP_DIR checkout-index -a -f --prefix=`pwd`/sltp

ls
ls sltp

cp sltp/images/Dockerfile .
docker build -t sltp .

# Upload image to the amazon cluster
docker save sltp | bzip2 | pv | ssh awscluster 'bunzip2 | docker load'

cleanup

popd