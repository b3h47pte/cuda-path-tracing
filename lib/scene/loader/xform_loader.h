#pragma once

#include "math/transform.h"
#include <json/json.hpp>

namespace cpt {

Transform load_xform_from_json(const nlohmann::json& jobj);

}
