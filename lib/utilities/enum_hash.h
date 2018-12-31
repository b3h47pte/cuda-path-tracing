#pragma once

namespace cpt {

// https://stackoverflow.com/questions/18837857/cant-use-enum-class-as-unordered-map-key
struct EnumHash
{
    template <typename T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

}
