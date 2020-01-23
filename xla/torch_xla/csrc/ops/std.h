#pragma once

#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Std : public Node {
 public:
  Std(const Value& input, std::vector<xla::int64> dimensions,
      bool keep_reduced_dimensions, bool unbiased);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<xla::int64>& dimensions() const { return dimensions_; }

  bool keep_reduced_dimensions() const { return keep_reduced_dimensions_; }

  bool unbiased() const { return unbiased_; }

 private:
  std::vector<xla::int64> dimensions_;
  bool keep_reduced_dimensions_;
  bool unbiased_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
