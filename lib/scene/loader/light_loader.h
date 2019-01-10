#pragma once

#include <json/json.hpp>
#include <memory>
#include "scene/lights/light.h"

namespace cpt {

class LightLoader
{
public:
    virtual ~LightLoader() = default;

    virtual LightPtr load_light_from_json(const nlohmann::json& jobj);

private:
    virtual LightPtr load_point_light_from_json(const nlohmann::json& jobj);
};

using LightLoaderPtr = std::shared_ptr<LightLoader>;

}
