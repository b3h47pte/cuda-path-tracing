#pragma once

#include <gmock/gmock.h>
#include <scene/loader/mesh_loader.h>

class MockMeshLoader: public cpt::MeshLoader
{
public:
    MOCK_METHOD1(load_mesh_from_file, cpt::GeometryPtr(const std::string&));
};
