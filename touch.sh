#!/bin/bash

# 检查是否有参数传入
if [ $# -eq 0 ]; then
    echo "使用方法: $0 '文件名1' '文件名2' '文件名3' ..."
    exit 1
fi

# 遍历所有参数
for file in "$@"
do
    # 检查文件是否已存在
    if [ -e "$file" ]; then
        echo "警告: 文件 '$file' 已存在，跳过创建。"
    else
        # 创建文件
        touch "$file"
        if [ $? -eq 0 ]; then
            echo "成功创建文件: $file"
        else
            echo "错误: 无法创建文件 '$file'"
        fi
    fi
done

echo "操作完成。"