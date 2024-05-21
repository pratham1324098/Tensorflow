
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_ADD_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_ADD_H_

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct AddOp {
  struct Attributes {};
};

AddOp Create(AddOp::Attributes);
absl::Status Prepare(AddOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output);
absl::Status Evaluate(AddOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_ADD_H_
