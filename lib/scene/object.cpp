#include "object.h"

namespace cpt {

Eigen::Vector3f Object::world_up_dir() const {
    return _object_to_world_xform * Eigen::Vector3f::UnitY();
}

Eigen::Vector3f Object::world_right_dir() const {
    return _object_to_world_xform * Eigen::Vector3f::UnitX();
}

Eigen::Vector3f Object::world_forward_dir() const {
    return _object_to_world_xform * -Eigen::Vector3f::UnitZ();
}

}
