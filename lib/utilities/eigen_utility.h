#pragma once

#include <Eigen/Core>
#include <vector>

namespace cpt {

template<typename T, int Dim>
Eigen::Array<T,Dim,Eigen::Dynamic> stl_vector_to_eigen_array(const std::vector<Eigen::Matrix<T, Dim, 1>>& vec)
{
    Eigen::Array<T,Dim,Eigen::Dynamic> arr;
    arr.resize(Eigen::NoChange, vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        arr.col(i) = vec[i];
    }
    return arr;
}

}
