#include "light_loader.h"
#include "scene/lights/point_light.h"
#include "utilities/error.h"
#include "utilities/json_utility.h"

namespace cpt {
namespace {

const std::string POINT_LIGHT = "point";

void load_common_light_properties_from_json(LightPtr& light, const nlohmann::json& jobj) {
    auto color_it = jobj.find("color");
    if (color_it != jobj.end()) {
        light->set_color(load_fixed_size_vector_from_array<float,3>(*color_it));
    }
}

}

LightPtr LightLoader::load_light_from_json(const nlohmann::json& jobj) {
    auto type_it = jobj.find("type");
    CHECK_AND_THROW_ERROR(type_it != jobj.end(), "No type specified for light.");

    if (*type_it == POINT_LIGHT) {
        return load_point_light_from_json(jobj);
    } else {
        THROW_ERROR("Invalid light type: " << *type_it);
    }
    
    return nullptr;
}

LightPtr LightLoader::load_point_light_from_json(const nlohmann::json& jobj) {
    LightPtr light = std::make_shared<PointLight>();
    load_common_light_properties_from_json(light, jobj);
    return light;
}


}
