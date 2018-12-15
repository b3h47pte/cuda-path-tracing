#pragma once

#include "scene/scene.h"
#include <string>

namespace cpt {

ScenePtr loadSceneFromJson(const std::string& fname);

}
