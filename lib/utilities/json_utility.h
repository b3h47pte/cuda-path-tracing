#pragma once

#include <json/json.hpp>

namespace cpt {

nlohmann::json load_json_from_file(const std::string& filename);

}
