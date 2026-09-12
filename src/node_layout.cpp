#include "node_layout.h"

#include <stdexcept>

namespace bsmp {

int NodeLayout::numNodes() const {
    return static_cast<int>(coordinates.size() / 3);
}

bool NodeLayout::isValid() const {
    if (coordinates.size() % 3 != 0) {
        return false;
    }
    const int nodes = numNodes();
    if (!mechanical_dof_indices.empty() && mechanical_dof_indices.size() != static_cast<size_t>(nodes * 3)) {
        return false;
    }
    if (!electrical_dof_indices.empty() && electrical_dof_indices.size() != static_cast<size_t>(nodes)) {
        return false;
    }
    return true;
}

int NodeLayout::mechanicalDofIndex(int node_index, int component) const {
    if (component < 0 || component >= 3) {
        throw std::out_of_range("Mechanical component must be 0, 1, or 2");
    }
    if (node_index < 0 || node_index >= numNodes()) {
        throw std::out_of_range("Node index out of range");
    }
    return mechanical_dof_indices[node_index * 3 + component];
}

int NodeLayout::electricalDofIndex(int node_index) const {
    if (node_index < 0 || node_index >= numNodes()) {
        throw std::out_of_range("Node index out of range");
    }
    if (electrical_dof_indices.empty()) {
        throw std::out_of_range("Electrical DOF map is empty");
    }
    return electrical_dof_indices[node_index];
}

NodeLayout NodeLayout::fromNodeCoordinates(const std::vector<float>& xyz_coordinates,
                                           bool include_electrical_dofs) {
    if (xyz_coordinates.size() % 3 != 0) {
        throw std::invalid_argument("Coordinate vector must contain triples (x, y, z)");
    }

    NodeLayout layout;
    layout.coordinates = xyz_coordinates;

    const int nodes = static_cast<int>(xyz_coordinates.size() / 3);
    layout.mechanical_dof_indices.resize(nodes * 3);
    for (int node = 0; node < nodes; ++node) {
        for (int component = 0; component < 3; ++component) {
            layout.mechanical_dof_indices[node * 3 + component] = node * 3 + component;
        }
    }

    if (include_electrical_dofs) {
        layout.electrical_dof_indices.resize(nodes);
        for (int node = 0; node < nodes; ++node) {
            layout.electrical_dof_indices[node] = node;
        }
    }

    return layout;
}

}  // namespace bsmp