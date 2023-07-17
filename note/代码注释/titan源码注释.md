## driver.py

用于深度学习框架测试的工具，会尝试运行在`args.input`指定的目录中找到的所有任务，并记录任务的执行结果。这个程序可以用于自动测试TensorFlow和Torch等框架，寻找可能的错误或者程序崩溃。

`clean_cache`

```
## 清理GPU缓存。通过在shell中运行命令来实现，该命令会找到所有属于用户'chenyuan'的、正在使用GPU设备/dev/nvidia*的Python进程，并杀死它们。
```



`main() -> int`

```
## 分析命令行参数，包括待测试的任务所在的文件夹，输出文件名，使用的框架（torch或tensorflow），测试模式（race或singleapi），是否继续上次的测试
## 如果没有指定从上次中断处继续运行，则重置捕获日志和跟踪日志
## 如果指定了从上次中断处继续运行，则从跟踪日志中读取当前进度
## 循环调用car，直到所有任务都被测试完
## 超时时cur自加，跳过当前任务，直到连续超时5次，然后跳过当前任务的下一个任务
## 监控输出文件的大小，如果长时间没有变化，则认为car进程卡住了，杀死进程
## 读取跟踪日志，获取当前已经测试的任务数量
```

## ev_generation.py

基于遗传算法的生成循环（generate loop）以生成符合特定要求的代码片段。

`generate_loop`

```
# 基于遗传算法的生成循环，使用SpanLM模型生成代码，使用GA类来选择种子
## 参数：
# args: 命令行参数
# model: SpanLM模型
# original_codes: 种子代码
# api: 目标API，生成的代码需要包含此API的调用
# logger: 日志记录器
# max_valid: 最大有效代码数量
```

## model.py

类名：SpanLM

`build_input`：输入函数

`build_input_multi`：创建多输入函数

`model_predict`：模型预测函数

## process_file.py

`clean_code(
    code: str,
    prints_and_imports=False,
    comment=False,
    cuda=False,
    fix_syntax=True,
    fix_session=True,
    remove_func=True,
) -> str:`：对代码进行清洗，去除标签。main函数、使解析容易出错的api等

`get_initial_seed_programs(directory: str, library: str, args) -> list`：从指定目录中读取所有的 Python 文件，将这些文件的元数据（API 名称、标签和原始代码）放入 tasks 列表中。如果指定了特定 API 或 ID，那么只会收集与这些参数匹配的文件。同时，如果在目录中存在一个名为 "outputs.json" 的文件，函数还会读取并生成报告"process.log"

`clean_programs(tasks, args) -> dict`：接受任务列表，对每一个任务进行处理：先使用clean_code函数清理原始代码，并将清理后的代码解析为 Python 的 AST。如果解析成功，会将清理后的代码以及其相关元数据添加到返回的字典中。