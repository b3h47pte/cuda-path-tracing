#pragma once

#include <Eigen/Core>
#include <vector>

namespace cpt {

template<typename T, int Dim>
Eigen::Array<T,Eigen::Dynamic,Dim> stl_vector_to_eigen_array(const std::vector<Eigen::Matrix<T, Dim, 1>>& vec)
{
    Eigen::Array<T,Eigen::Dynamic,Dim> arr;
    arr.resize(vec.size(), Eigen::NoChange);
    for (size_t i = 0; i < vec.size(); ++i) {
        arr.row(i) = vec[i];
    }
    return arr;
}

}
