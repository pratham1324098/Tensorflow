#include "tensorflow/lite/experimental/shlo/ops/add.h"
#include<functional>
#include<string>
#include<gmock/gmock.h>
#include<gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise_test_util.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"


using testing::FloatEq;
using testing::Pointwise;
namespace shlo_ref{
    template<>
    struct ParamName<AddOp>{
        static std::string Get() {return "Add";}
    };
    struct Add:std::plus<void>{};
    namespace{
        INSTANTIATE_TYPED_TEST_SUITE_P(Add,
                                        BinaryElementwiseOpShapePropagationTest,
                                        AddOp, TestParamNames);
        using MultipyBaselineContraintTypes = BinaryElementwiseBaselineConstraintTypes<
            AddOp,
            ConcatTypes<BaselineConstraintIntTypes, BaselineConstraintFloatTypes,
            BaselineConstraintQuantizedPerTensorTypes>>;
        INSTANTIATE_TYPED_TEST_SUITE_P(
            Add, BinaryElementwiseSameBaselineElementTypeConstraintTest,
            MultipyBaselineContraintTypes, TestParamNames);
        using UnsupportedTypes =
            WithOpTypes<AddOp,
                        ConcatTypes<BoolTestType, PerAxisQuantizedTestTypes>>;
        INSTANTIATE_TYPED_TEST_SUITE_P(Add, BinaryElementwiseUnsupportedTypeTest,
                                        UnsupportedTypes, TestParamNames);
        using ArithmeticTypes = ConcatTypes<ArithmeticTestTypes>;
        template<class T>
        struct AddTest: ::testing::Test {};
                
        TYPED_TEST_SUITE(AddTest, ArithmeticTypes, TestParamNames);

        TYPED_TEST(AddTest, ArithmeticTestTypesTensorsWork) {
            using StorageT = typename TypeParam::StorageT;
            const Shape shape({2, 3, 4});
            Vector<StorageT> lhs_data =
                RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-50, /*max=*/50);
            Vector<StorageT> rhs_data =
                RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-5, /*max=*/5);
            Vector<StorageT> output_data(shape.NumElements());
            Tensor lhs_tensor{
                .type = TensorType{.shape=shape,.element_type=TypeParam::kStorage},
                .data = lhs_data.data()};
            Tensor rhs_tensor{
                .type = TensorType{.shape=shape,.element_type=TypeParam::kStorage},
                .data = rhs_data.data()};
            Tensor output_tensor{
                .type = TensorType{.shape=shape,.element_type=TypeParam::kStorage},
                .data = output_data.data()};
            Vector<StorageT> expected_data(shape.NumElements());
            absl::c_transform(lhs_data,rhs_data,expected_data.begin(),Add());
            auto op = Create(AddOp::Attributes{});
            ASSERT_OK(Prepare(op,lhs_tensor,rhs_tensor,output_tensor));
            ASSERT_OK(Evaluate(op,lhs_tensor,rhs_tensor,output_tensor));
            EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));

        }
        template <class T>
        struct QuantizedAddTest : ::testing::Test {};
        TYPED_TEST_SUITE(QuantizedAddTest, QuantizedTestTypes, TestParamNames);
        TYPED_TEST(QuantizedAddTest, PerTensorWorks) {
        using StorageT = typename TypeParam::StorageT;
        using ExpressedT = typename TypeParam::ExpressedT;
        const Shape shape({2,3,4});
        const ExpressedT scale =static_cast<ExpressedT>(1.5);
        const StorageT zero_point = static_cast<StorageT>(2);
        Vector<StorageT> lhs_data =
            RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-50, /*max=*/50);
        Vector<StorageT> rhs_data =
            RandomBuffer<TypeParam::kStorage>(shape, /*min=*/-5, /*max=*/5);
        Vector<StorageT> output_data(shape.NumElements());
        const QuantizedElementTypePerTensor tensor_type =
            QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                          TypeParam::kExpressed, scale);
        Tensor lhs_tensor{
            .type = QuantizedPerTensorTensorType{.shape = shape,
                                           .element_type = tensor_type},
            .data = lhs_data.data()};
        Tensor rhs_tensor{
            .type = QuantizedPerTensorTensorType{.shape = shape,
                                           .element_type = tensor_type},
            .data = rhs_data.data()};
        Tensor output_tensor{
            .type = QuantizedPerTensorTensorType{.shape = shape,
                                           .element_type = tensor_type},
            .data = output_data.data()};
        Vector<StorageT> expected_data(shape.NumElements());
        absl::c_transform(
            lhs_data, rhs_data, expected_data.begin(),
            [zero_point, scale](auto lhs, auto rhs) {
                const ExpressedT dequantized_lhs = Dequantize(lhs, zero_point, scale);
                const ExpressedT dequantized_rhs = Dequantize(rhs, zero_point, scale);
                const ExpressedT dequantized_res =
                    Add()(dequantized_lhs, dequantized_rhs);
                return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
                    dequantized_res, zero_point, static_cast<ExpressedT>(1.) / scale);
            });
        auto op = Create(AddOp::Attributes{});
        ASSERT_OK(Prepare(op, lhs_tensor, rhs_tensor, output_tensor));
        ASSERT_OK(Evaluate(op, lhs_tensor, rhs_tensor, output_tensor));
        EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
        }
    }
    
}