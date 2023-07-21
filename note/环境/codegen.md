# Codegen

```
pip install transformers==4.29.2
pip install tiktoken==0.4.0
# 太慢可加 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host https://pypi.tuna.tsinghua.edu.cn
```

需要PyTorch、TensorFlow或者Flax中任意框架才能正常启动
```
pip install tensorflow 
```

进入python终端后输入，文件运行会报错
```
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-mono", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen25-7b-mono")
inputs = tokenizer("def hello_world():", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0]))
```

跑完后结果如下

![Alt text](../../../assets/image-3.png)

用显卡跑，试了很久，总结下来要做以下几步
安装Cuda，安装CudaNN，安装带显卡的torch(一般的torch不带显卡)

安装CudaNN https://developer.nvidia.com/rdp/cudnn-download
安装带显卡的torch https://pytorch.org/get-started/locally/  （他给的几个版本都没有12.2，先装个11.8的试试，pip的装不上，用conda的）
![Alt text](../../../assets/image-4.png)


```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.2/extras/CUPTI/lib64
```

确认有GPU

```
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```


使用以下命令检测显卡

```bash
watch -n 5 nvidia-smi
```

在上面的代码中加上

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```


下载报错
```
>>> model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen25-7b-mono")
Downloading shards:   0%|                                                                                                                                 | 0/3 [00:02<?, ?it/s]
Downloading (…)l-00001-of-00003.bin:   0%|▏                                                                                              | 21.0M/9.95G [00:10<1:20:30, 2.05MB/s]
Traceback (most recent call last):n:   0%|                                                                                                 | 10.5M/9.95G [00:01<21:00, 7.88MB/s] 
  File "<stdin>", line 1, in <module>
  File "C:\Users\cc241\AppData\Roaming\Python\Python310\site-packages\transformers\models\auto\auto_factory.py", line 467, in from_pretrained
    return model_class.from_pretrained(
  File "C:\Users\cc241\AppData\Roaming\Python\Python310\site-packages\transformers\modeling_utils.py", line 2523, in from_pretrained
    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
  File "C:\Users\cc241\AppData\Roaming\Python\Python310\site-packages\transformers\utils\hub.py", line 934, in get_checkpoint_shard_files
    cached_filename = cached_file(
  File "C:\Users\cc241\AppData\Roaming\Python\Python310\site-packages\transformers\utils\hub.py", line 417, in cached_file
    resolved_file = hf_hub_download(
  File "C:\Users\cc241\AppData\Roaming\Python\Python310\site-packages\huggingface_hub\utils\_validators.py", line 120, in _inner_fn
    return fn(*args, **kwargs)
  File "C:\Users\cc241\AppData\Roaming\Python\Python310\site-packages\huggingface_hub\file_download.py", line 1364, in hf_hub_download
    http_get(
  File "C:\Users\cc241\AppData\Roaming\Python\Python310\site-packages\huggingface_hub\file_download.py", line 544, in http_get
    temp_file.write(chunk)
  File "D:\ProgramData\Anaconda3\lib\tempfile.py", line 483, in func_wrapper
    return func(*args, **kwargs)
OSError: [Errno 28] No space left on device
```

一看发现C盘满了，我将conda的环境从C转移到了D，再次运行仍然报错，查询得知是由于transformers库默认将预训练模型下载到用户的主目录（此时是在windows上运行）

```
set TRANSFORMERS_CACHE=D:\environment\transformers # 这是临时设置，永久设置需要添加环境变量TRANSFORMERS_CACHE
```

接下来进行了很长时间的下载

![Alt text](../../../assets/image-2.png)

在这期间使用linux装了个，遇到几个报错，需要安装这些东西

```
pip install sentencepiece
pip install torch # 一开始没加后来model加载不出来提示要torch
```

## 重配

大佬告诉我这个模型是7b的，很慢，换小的跑

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-mono")
inputs = tokenizer("# this function prints hello world", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))
```