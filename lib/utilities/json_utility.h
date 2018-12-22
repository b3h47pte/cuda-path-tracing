#pragma once

#include <Eigen/Core>
#include <json/json.hpp>
#include "utilities/error.h"

namespace cpt {

nlohmann::json load_json_from_file(const std::string& filename);

template<typename T, int D>
Eigen::Matrix<T,D,1> load_fixed_size_vector_from_array(const nlohmann::json& jobj) {
    CHECK_AND_THROW_ERROR(jobj.is_array(), "JSON object is not an array.");
    CHECK_AND_THROW_ERROR(jobj.size() == D, "JSON array dimensions not equal to fixed size.");

    Eigen::Matrix<T,D,1> ret;
    for (auto i = 0; i < D; ++i)
        ret(i) = jobj[i].get<T>();
    return ret;
}

}
