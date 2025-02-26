#!/bin/bash
echo "[Ascend310B1] Generating ArgMaxWithValue_ef0fb1765342d1cb1f3aece201b5dc90 ..."
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
res=$(opc $1 --main_func=arg_max_with_value --input_param=/home/HwHiAiUser/main/build_out/op_kernel/binary/ascend310b/gen/ArgMaxWithValue_ef0fb1765342d1cb1f3aece201b5dc90_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ArgMaxWithValue_ef0fb1765342d1cb1f3aece201b5dc90.json ; then
  echo "$2/ArgMaxWithValue_ef0fb1765342d1cb1f3aece201b5dc90.json not generated!"
  exit 1
fi

if ! test -f $2/ArgMaxWithValue_ef0fb1765342d1cb1f3aece201b5dc90.o ; then
  echo "$2/ArgMaxWithValue_ef0fb1765342d1cb1f3aece201b5dc90.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating ArgMaxWithValue_ef0fb1765342d1cb1f3aece201b5dc90 Done"
