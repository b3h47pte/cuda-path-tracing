#pragma once

#define CUDA_MAX_STACK_SIZE 128

namespace cpt {

template<typename T>
class CudaStack
{
public:
    CUDA_DEVHOST void push(const T& obj) {
        const bool local_oob = (_current_idx + 1) >= CUDA_MAX_STACK_SIZE;
        _out_of_bounds |= local_oob;
        if (!local_oob) {
            _stack[++_current_idx] = obj;
        }
    }

    CUDA_DEVHOST T pop() {
        if (!empty()) {
            --_current_idx;
        }
        return _stack[_current_idx + 1];
    }

    CUDA_DEVHOST bool empty() const {
        return (_current_idx == -1);
    }

    CUDA_DEVHOST bool out_of_bounds() const {
        return _out_of_bounds;
    }

private:
    T _stack[CUDA_MAX_STACK_SIZE];
    int _current_idx{-1};

    // So we can detecet if we ever push out of bounds...
    bool _out_of_bounds{false};
};

}
