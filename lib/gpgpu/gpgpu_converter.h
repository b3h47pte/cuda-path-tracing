#pragma once

#include <unordered_map>
#include <tbb/concurrent_unordered_map.h>

namespace cpt {

class Triangle;
class GeometryAggregate;
class PinholePerspectiveCamera;

class PointLight;

class GpgpuConverter
{
public:
    virtual ~GpgpuConverter() = default;

    // Geometry
    virtual void convert(const Triangle& triangle) = 0;
    virtual void convert(const GeometryAggregate& aggregate) = 0;
    
    // Cameras
    virtual void convert(const PinholePerspectiveCamera& camera) = 0;

    // Lights
    virtual void convert(const PointLight& light) = 0;

    void add_to_cache(void* key, void* value) {
        _cache[key] = value;
    }

    template<typename T>
    T* get_from_cache(void* key) const {
        auto it = _cache.find(key);
        if (it == _cache.end()) {
            return nullptr;
        }
        return static_cast<T*>(it->second);
    }

private:
    tbb::concurrent_unordered_map<void*,void*> _cache;
};

}
