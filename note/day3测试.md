# day3

## libxml2

寻找`CVE-2015-8317`

题目链接https://github.com/mykter/afl-training/tree/main/challenges/libxml2

先了解下libxml2的基本使用





#### ASAN

参考文章https://zhuanlan.zhihu.com/p/51443698，http://blog.binpang.me/2018/07/31/afl-asan/

> AddressSanitizer，高效的内存错误检测工具，原先是LLVM的特性，后被加入GCC4.9，使用方法是在AFL编译软件前设置环境变量AFL_USE_ASAN=1 
>
> ```bash
> export AFL_USE_ASAN=1
> ```
>
> **为什么要开启ASAN**
>
> 1. 很多内存操作的错误不会导致程序崩溃，例如越界，不开启ASAN很多内存错误无法被AFL发现
>
> **注意事项**
>
> 由于ASAN工具会跟踪所有内存，所以需要的内存很大，分析32位程序中最多占用800MB内存；而分析64位程序时会占用20TB，因此需要作出相应更改：
>
> 实际上，以上最大内存只是理论上的，一般运行的程序shadow memory所占用的内存并没有这么多，所以第一种解决方法就是使用-m none选项，来忽略此错误:
>
> ```
> afl-fuzz -i in -o out -m none ./executable
> ```
>
> 第二种方法就是使用cgroup来限定改程序使用的资源：
>
> ```
> sudo ~/afl/experimental/asan_cgroups/limit_memory.sh -u usename afl-fuzz -i in -o out -m none ./executable
> ```
>
> 第二种方法是比较稳妥的方法，并不会对系统造成非常大的影响，因为其限定了程序所使用的内存资源。

### 开始复现

发现可以通过docker进项服务的安装

```bash
cd environment
sudo docker build . -t fuzz-training
```

