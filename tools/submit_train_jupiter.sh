#!/bin/bash
GPUS=$1
CONFIG=$2
MODEL=$3
PY_ARGS=${@:4}

MASTER_PORT=29500
N=1
DIST="-m"
DIST_ARGS=""
if [ ${GPUS} -gt 8 ]
then
    echo "multi machine"
    N=$((${GPUS} / 8))
    DIST="-d AllReduce"
    GPUS=8
    DIST_ARGS="--nnodes=\${WORLD_SIZE} --node_rank=\${RANK} --master_addr=\${MASTER_ADDR}"
fi

PART='jupiter'

if [ ${PART} = "jupiter" ]
then
    PID='c24b2f96-bb5d-4eb5-b1f6-e33733686228'
    RID='N1lS.Ia.I20'
elif [ ${PART} = 'lg' ]
then
    PID='b377fa28-094f-40ca-aabe-ad95597273ad'
    RID='N1lS.Ib.I00'
else
    PID='da6b6a11-1bb7-4316-ad28-ff9f00c9ebb6'
    RID='N5IP.nn.I90'
fi

set -x

srun \
    -j train \
    -p ${PID} \
    --workspace-id e8adf4f9-16e7-4cfc-b65e-cad12d84c108 \
    -r ${RID}.${GPUS} \
    -f PyTorch \
    ${DIST} \
    -N ${N} \
    -o ./train.log \
    --container-image registry.st-sh-01.sensecore.cn/cabinrd-ccr/huangtao-pt1p12-env-20231114:20231114-17h15m55s \
    --container-mounts a79904e9-72fe-11ee-903c-4a906bc4e079:/mnt/afs \
    bash -c \
    "cd ${PWD} && source activate cloud-ai-lab && MASTER_PORT=$MASTER_PORT PYTHONWARNINGS=ignore PYTHONPATH=$PWD:$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=${GPUS} ${DIST_ARGS} tools/train.py -c ${CONFIG} --model ${MODEL} ${PY_ARGS}"

