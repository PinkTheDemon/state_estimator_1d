#!/bin/bash

# 定义参数列表
cov_list=(
    "1e-4"
    "1e-2"
    "1e0"
    "1e2"
    "1e4"
)
goodInit_list=(
    # ""
    "True"
)
gamma_list=(
    0.4
    0.6
    0.8
    0.99
    1.0
)

# 循环执行Python脚本
for goodInit in "${goodInit_list[@]}"
do
    for gamma in "${gamma_list[@]}"
    do 
        for cov in "${cov_list[@]}"
        do
            python ./linear.py --cov $cov --gamma $gamma # --goodInit $goodInit
        done
    done
done