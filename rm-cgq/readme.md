# Remote Reward Server 发车指南

## 目前端口对应情况


| 部署情况 | 接收端口 | xverify端口 | gpu使用数 | 主要使用者 | 
|----------|----------|-------------|------------|----|
| rule + xverify 适配slime  | 8000 | 34881 | 2 | cjc |
| rule + xverify 适配slime  | 8001 | 34882 | 2 | cgq |
| xverify 适配verl  | 34883 | 34883 | 1 | ly |
| rule + xverify 适配slime  | 8003 | 34884 | 1 | cgq |

## 指定节点
目前使用的节点：
```
--positive-tags node/gpu-lg-cmc-h-h200-1005.host.h.pjlab.org.cn \
```


## 启动
```
bash run_rm.sh -m 34885 -r 8005
```