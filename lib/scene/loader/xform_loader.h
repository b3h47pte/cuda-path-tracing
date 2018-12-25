#pragma once

#include "math/transform.h"
#include <memory>
#include <json/json.hpp>

namespace cpt {

class XformLoader
{
public:
    virtual ~XformLoader() = default;

    virtual Transform load_xform_from_json(const nlohmann::json& jobj);
};

using XformLoaderPtr = std::shared_ptr<XformLoader>;

}
