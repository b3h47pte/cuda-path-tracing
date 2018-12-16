#pragma once

#include <memory>

namespace cpt {

class Object
{
public:
    virtual ~Object() {}
};

using ObjectPtr = std::shared_ptr<Object>;

}
