#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ArgMaxWithValue)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(indice, ge::TensorType::ALL())
    .OUTPUT(values, ge::TensorType::ALL())
    .REQUIRED_ATTR(dimension, Int)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ArgMaxWithValue);

}

#endif
