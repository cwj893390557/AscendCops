#!/bin/bash
echo "[Ascend310B1] Generating ArgMaxWithValue_3b57390fa9b18c0f3eea66cebdf80fa1 ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=arg_max_with_value --input_param=/home/HwHiAiUser/main/build_out/op_kernel/binary/ascend310b/gen/ArgMaxWithValue_3b57390fa9b18c0f3eea66cebdf80fa1_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ArgMaxWithValue_3b57390fa9b18c0f3eea66cebdf80fa1.json ; then
  echo "$2/ArgMaxWithValue_3b57390fa9b18c0f3eea66cebdf80fa1.json not generated!"
  exit 1
fi

if ! test -f $2/ArgMaxWithValue_3b57390fa9b18c0f3eea66cebdf80fa1.o ; then
  echo "$2/ArgMaxWithValue_3b57390fa9b18c0f3eea66cebdf80fa1.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating ArgMaxWithValue_3b57390fa9b18c0f3eea66cebdf80fa1 Done"
