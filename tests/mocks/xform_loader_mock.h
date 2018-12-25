#pragma once

#include <gmock/gmock.h>
#include <scene/loader/xform_loader.h>

class MockXformLoader: public cpt::XformLoader
{
public:
    MOCK_METHOD1(load_xform_from_json, cpt::Transform(const nlohmann::json&));
};
