
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ARG_MAX_WITH_VALUE_H_
#define ACLNN_ARG_MAX_WITH_VALUE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnArgMaxWithValueGetWorkspaceSize
 * parameters :
 * x : required
 * dimension : required
 * keepDimsOptional : optional
 * indiceOut : required
 * valuesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnArgMaxWithValueGetWorkspaceSize(
    const aclTensor *x,
    int64_t dimension,
    bool keepDimsOptional,
    const aclTensor *indiceOut,
    const aclTensor *valuesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnArgMaxWithValue
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnArgMaxWithValue(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
