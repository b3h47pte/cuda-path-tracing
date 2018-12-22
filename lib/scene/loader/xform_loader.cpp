#include "xform_loader.h"

#include "math/rotation.h"
#include "utilities/error.h"
#include "utilities/json_utility.h"

namespace cpt {
namespace {

Eigen::Matrix3f load_rotation_matrix_from_type(const std::string& type, const nlohmann::json& jobj) {
    if (type == "euler_xyz") {
        return get_euler_xyz_rotation_matrix(load_fixed_size_vector_from_array<float,3>(jobj));
    } else {
        THROW_ERROR("Unsupported rotation type: " << type);
    }
    return Eigen::Matrix3f();
}

}

Transform load_xform_from_json(const nlohmann::json& jobj) {
    Transform xform;

    auto translation_it = jobj.find("translation");
    if (translation_it != jobj.end()) {
        xform.set_translation(load_fixed_size_vector_from_array<float,3>(*translation_it));
    }

    auto scale_it = jobj.find("scale");
    if (scale_it != jobj.end()) {
        xform.set_scale(load_fixed_size_vector_from_array<float,3>(*scale_it));
    }

    auto rotation_it = jobj.find("rotation");
    if (rotation_it != jobj.end()) {
        auto rotation_type_it = jobj.find("rotation_type");
        CHECK_AND_THROW_ERROR(rotation_type_it != jobj.end(), "If rotation for a transform is specified, its 'rotation_type' must also be specified.");
        xform.set_rotation(load_rotation_matrix_from_type(*rotation_type_it, *rotation_it));
    }

    return xform;
}

}
