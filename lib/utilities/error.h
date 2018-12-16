#pragma once

#include <stdexcept>
#include <sstream>
#include <iostream>

#define THROW_ERROR(x) \
    {\
        std::stringstream ss;\
        ss << x;\
        throw std::runtime_error(ss.str());\
    }

#define CHECK_AND_THROW_ERROR(c, x) \
    if (!(c)) {\
        THROW_ERROR(x); \
    }
