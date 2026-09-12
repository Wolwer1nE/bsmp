#ifndef BSMP_NODE_LAYOUT_H
#define BSMP_NODE_LAYOUT_H

#include <vector>

namespace bsmp {

struct NodeLayout {
    std::vector<float> coordinates;
    std::vector<int> mechanical_dof_indices;
    std::vector<int> electrical_dof_indices;

    int numNodes() const;
    bool isValid() const;

    int mechanicalDofIndex(int node_index, int component) const;
    int electricalDofIndex(int node_index) const;

    static NodeLayout fromNodeCoordinates(const std::vector<float>& xyz_coordinates,
                                          bool include_electrical_dofs = true);
};

}  // namespace bsmp

#endif  // BSMP_NODE_LAYOUT_H