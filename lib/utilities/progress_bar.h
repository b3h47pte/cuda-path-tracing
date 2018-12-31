#pragma once

#include <atomic>
#include <cstddef>
#include <thread>

namespace cpt {

class ProgressBar
{
public:
    ProgressBar(size_t total);

    void job_done();
    void complete();

private:
    std::atomic<size_t> _completed{0};
    size_t _total;
    bool _is_done{false};
    std::thread _worker_thread;
};

}
