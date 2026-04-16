#ifndef BSMP_ELASTIC_NULLSPACE_H
#define BSMP_ELASTIC_NULLSPACE_H

#include <vector>

#include "node_layout.h"

namespace bsmp {

std::vector<std::vector<float>> build_rigid_body_modes(const NodeLayout& layout);

}  // namespace bsmp

#endif  // BSMP_ELASTIC_NULLSPACE_H