#include "progress_bar.h"
#include <chrono>
#include <iostream>

namespace cpt {

ProgressBar::ProgressBar(size_t total):
    _total(total) {
    _worker_thread = std::thread([this](){
        while (!_is_done) {
            std::cout << "Progress: " << _completed << "/" << _total << "(" << static_cast<double>(_completed) / _total * 100.0 << "%)\r" << std::flush;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });
}

void ProgressBar::job_done() {
    ++_completed;
}

void ProgressBar::complete() {
    _is_done = true;
    _worker_thread.join();
    std::cout << std::endl;
}

}
