### S3性能比赛代码
#### 路径\架构切换
```bash
npu-smi info #查看昇腾架构
```
/op_host/arg_max_with_value.cpp:
```
this->AICore().AddConfig("") #切换成开发板架构
```
CMakePresets.json:
```
"value": "/usr/local/Ascend/ascend-toolkit/latest" #换成算子包路径
```

#### Example
```bash
bash build.sh
cd build_out
./custom_opp_openEuler_aarch64.run
```
#### Test
```bash
cd /path/to/ArgWithMax
bash run.sh
```
