#include "tensorflow/lite/experimental/shlo/ops/add.h"
#include<functionl>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"


namespace shlo_ref {
    struct Add : std::plus<void>{};
    AddOp Create(AddOp::Attributes) {return {}; }

    absl::Status Prepare(AddOp& op, const Tensor &lhs, const Tensor &rhs, Tensor&output){
        SHLO_REF_RETURN_ON_ERROR(Propagate(lhs.shape(), rhs.shape(), output.shape()));
        SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(CheckCtx("add"), lhs,
                                               IsIntTensor, IsFloatTensor,
                                               IsQuantizedPerTensorTensor));
        SHLO_REF_RETURN_ON_ERROR(
            CheckSameBaselineType(CheckCtx("add"), lhs, output));
        SHLO_REF_RETURN_ON_ERROR(
            CheckSameBaselineType(CheckCtx("add"), rhs, output));
        return absl::OkStatus();
    }
    absl::Status Evaluate(AddOp& op, const Tensor&lhs, const Tensor& rhs, Tensor& output){
        Add add;
        if(IsIntTensor(lhs)||isFloatTensor(lhs)){
            DISPATCH_INT_FLOAT(detail::EvaluateNoQuantization, lhs.tensor_element_type(),add,lhs,rhs,output);
        }else if(IsQuantizedPerTensorTensor(lhs)){
            DISPATCH_QUANTIZED(detail::DequantizeOpQuantizePerTensor,
                                lhs.quantized_per_tensor_element_type().StorageType(),
                                lhs.quantized_per_tensor_element_type().ExpressedType(),
                                add,lhs,rhs,output)
        }
        return absl::FailedPreconditionError(
            "stablehlo.add: Unsupported tensor type.");
    }
}