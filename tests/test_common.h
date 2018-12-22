#pragma once

#include <Eigen/Core>
#include <scene/geometry/geometry.h>
#include <iostream>

// TODO: Rename..
extern Eigen::Array3Xf ref_positions;
extern Eigen::Array2Xf ref_uvs;
extern Eigen::Array3Xf ref_normals;

void check_test_plane(const cpt::GeometryPtr& geom);
cpt::GeometryPtr construct_test_plane();
