t5s中文语言模型
===========================

语料：[nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)

训练平台：Colab [白嫖Colab训练语言模型教程](https://github.com/genggui001/everyone_can_pretrain_language_model)

基础框架：苏神的[bert4keras](https://github.com/bojone/bert4keras)

框架安装：

```
pip install bert4keras==0.9.9
```

模型代码：

[chinese_t5s.py](chinese_t5s.py)

## base版本

### 模型参数下载地址：

百度网盘：[链接](https://pan.baidu.com/s/1SLc_0yfEAukdGB2ctfEgTQ) 提取码：kqiu

### 模型配置：

```
{
    "attention_head_size": 64, 
    "hidden_act": [
        "gelu", 
        "linear"
    ], 
    "hidden_dropout_prob": 0.1, 
    "hidden_size": 768, 
    "initializer_range": 0.02, 
    "intermediate_size": 2048, 
    "num_attention_heads": 12, 
    "num_hidden_layers": 12, 
    "type_vocab_size": 3, 
    "vocab_size": 13686
}
```

### 模型预训练过程：

两阶段预训练：90%采用128句子长度，10%采用512句子长度

128阶段预训练参数：

```
sequence_length = 128
batch_size = 4096
learning_rate = 5 /  (pow(2, 1.5) * 1e3)
weight_decay_rate = 0.01
num_warmup_steps = 0
num_train_steps = 250000
```

mlm_loss曲线：

![chinese_t5s_base_mlm_loss_step_0](images/chinese_t5s_base_mlm_loss_step_0.png)

mlm_acc曲线：

![chinese_t5s_base_mlm_acc_step_0](images/chinese_t5s_base_mlm_acc_step_0.png)

512阶段预训练参数：

```
sequence_length = 512
batch_size = 4096
learning_rate = 5 /  (pow(2, 1.5) * 1e3) / 10
weight_decay_rate = 0.01
num_warmup_steps = 0
num_train_steps = 25000
```

mlm_loss曲线：

整理中

mlm_acc曲线：

整理中


## Tiny No Dropout版本

### 模型参数下载地址：

百度网盘：[链接](https://pan.baidu.com/s/1mWZUuEbCb5nbWRdN1TdnKw) 提取码：shz8

### 模型配置：

```
{
    "attention_head_size": 32, 
    "hidden_act": [
        "gelu", 
        "linear"
    ],
    "hidden_dropout_prob": 0,
    "dropout_rate": 0,
    "hidden_size": 384,
    "embedding_size": 128,
    "initializer_range": 0.02, 
    "intermediate_size": 1024, 
    "num_attention_heads": 12, 
    "num_hidden_layers": 4, 
    "type_vocab_size": 3, 
    "vocab_size": 13686
}
```
### 模型预训练过程：

两阶段预训练：90%采用128句子长度，10%采用512句子长度

128阶段预训练参数：

```
sequence_length = 128
batch_size = 4096
learning_rate = 5 /  (pow(2, 1.5) * 1e3)
weight_decay_rate = 0.01
num_warmup_steps = 0
num_train_steps = 250000
```

mlm_loss曲线：

![chinese_t5s_tiny_mlm_loss_step_0](images/chinese_t5s_tiny_mlm_loss_step_0.png)

mlm_acc曲线：

![chinese_t5s_tiny_mlm_acc_step_0](images/chinese_t5s_tiny_mlm_acc_step_0.png)

512阶段预训练参数：

```
sequence_length = 512
batch_size = 4096
learning_rate = 5 /  (pow(2, 1.5) * 1e3) / 10
weight_decay_rate = 0.01
num_warmup_steps = 0
num_train_steps = 25000
```

mlm_loss曲线：

![chinese_t5s_tiny_mlm_loss_step_1](images/chinese_t5s_tiny_mlm_loss_step_1.png)

mlm_acc曲线：

![chinese_t5s_tiny_mlm_acc_step_1](images/chinese_t5s_tiny_mlm_acc_step_1.png)

