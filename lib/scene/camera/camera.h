#pragma once

#include "scene/object.h"

namespace cpt {

class Camera : public Object
{

};

using CameraPtr = std::shared_ptr<Camera>;

}
