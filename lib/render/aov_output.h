#pragma once

#include <boost/gil.hpp>
#include <string>
#include <unordered_map>
#include "utilities/enum_hash.h"

namespace cpt {

class AovOutput
{
public:

    enum class Channels
    {
        FinalImage
    };

    void initialize(size_t width, size_t height);
    void add_channel(Channels channel);
    void save_channel_to_file(Channels channel, const std::string& filename) const;

private:
    size_t _width{0};
    size_t _height{0};

    std::unordered_map<Channels, boost::gil::rgb32_image_t, EnumHash> _aovs;
};

}