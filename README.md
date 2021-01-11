# NGAT4Rec
NGAT4Rec: Neighbor-Aware Graph Attention Network For Recommendation

* hgm
  - Data
    + 数据
      * kuaishou(10-core):
        * 快手公共数据集，用户：68257，视频：185963
      * gowalla(10-core):
        * 用户：29858，商品：40981
      * yelp2018(10-core):
        * 用户：45919，商品：45538
      * amazon-book(10-core):
        * 用户：52643，商品：91599
  - DGLModels
    + SubGraphModels
      * 模型代码
      * utility
        - 数据集预处理等
      * evaluator
        - 评估

* 环境：
  - Python 3.6
  - PyTorch 1.5
  - DGL 0.4.3

运行：
```python NGAT.py --dataset amazon-book --regs [1e-5] --gpu_id 1 --embed_size 64 --layer_size [64] --lr 0.0001 --save_flag 1 --pretrain 0 --neighbors_num [64,32] --batch_size 1024 --epoch 2000```

* 参数解释
  - --dataset: 数据集名称，可选择`kwai`、`gowalla`、`yelp2018`或`amazon-book`
  - --gpu_id: GPU编号，训练前记得输入`nvidia-smi`确认GPU空闲情况，否则GPU会报OOM
  - --embed_size: 初始查词矩阵embedding维度
  - --save_flag: 保存开关，0为关闭，1为开启
  - --batch_size: mini-batch大小，一般为1024
  - --layer_size: 神经网络每一层的维度
  - --neighbors_num: 采样邻居数目与深度，例如[64,32]，即采样64个1跳邻居，再采样每个1跳邻居的32个邻居即64*32个2跳邻居
  - --epoch：训练轮数

utility和evaluate的部分代码来自LightGCN https://github.com/kuandeng/LightGCN.git，感谢
