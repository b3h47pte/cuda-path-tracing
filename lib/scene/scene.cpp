#include "scene.h"

namespace cpt {

Scene::Scene(std::vector<GeometryPtr>&& geometry):
    _geometry(std::move(geometry))
{
}

}
