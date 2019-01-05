#include "aov_output.h"
#include <boost/gil/io/io.hpp>
#include <boost/gil/extension/io/tiff.hpp>
#include "utilities/error.h"

namespace cpt {

void AovOutput::initialize(size_t width, size_t height) {
    _width = width;
    _height = height;
    _aovs.clear();
    add_channel(Channels::FinalImage);
}

void AovOutput::add_channel(Channels channel) {
    CHECK_AND_THROW_ERROR(_width > 0 && _height > 0, "Need to initialize AovOutput before adding channel.");
    switch (channel) {
    case Channels::FinalImage:
        _aovs[channel] = boost::gil::rgb32f_image_t(_width, _height);
        boost::gil::fill_pixels(boost::gil::view(_aovs[channel]), boost::gil::rgb32f_image_t::value_type());
        break;
    default:
        THROW_ERROR("Unsupported channel type.");
        break;
    }
}

void AovOutput::save_channel_to_file(Channels channel, const std::string& filename) const {
    auto it = _aovs.find(channel);
    CHECK_AND_THROW_ERROR(it != _aovs.end(), "Can not save an uninitialized channel.");
    boost::gil::write_view(filename, boost::gil::const_view(it->second), boost::gil::tiff_tag());
}

std::vector<AovOutput::Channels> AovOutput::active_channels() const {
    std::vector<AovOutput::Channels> channels;
    channels.clear();
    channels.reserve(_aovs.size());
    for (const auto& kvp : _aovs) {
        channels.push_back(kvp.first);
    }
    return channels;
}

const boost::gil::rgb32f_image_t& AovOutput::image(Channels channel) const {
    return _aovs.at(channel);
}

boost::gil::rgb32f_image_t& AovOutput::image(Channels channel) {
    return _aovs[channel];
}

}
