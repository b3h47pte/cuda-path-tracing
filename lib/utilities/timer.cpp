#include "timer.h"
#include <iostream>

namespace cpt {

Timer::Timer(const std::string& msg):
    _msg(msg) {
}

void Timer::start() {
    _last_start = Clock::now();
    std::cout << "START: " << _msg << std::endl;
}

void Timer::tick() {
    auto now = Clock::now();
    auto diff = now - _last_start;
    std::cout << "END: " << _msg << "[" << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() / 1000.0  << "s]"<< std::endl;
    _last_start = now;
}

}
