#pragma once

#include "gpgpu/cuda_utils.h"

namespace cpt {

// Function have type bool operator()(const T&) and should
// return true if the object T should be kept.
template<typename T, typename Function>
T* cuda_stream_compact(T* start, T* end, Function f)
{
    // TODO:
    return end;
}

}
