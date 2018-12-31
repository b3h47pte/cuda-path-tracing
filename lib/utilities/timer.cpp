#include "timer.h"

namespace cpt {

Timer::Timer(
    const std::string& msg,
    LogLevel level):
    _msg(msg),
    _level(level) {
}

void Timer::start() {
    _last_start = Clock::now();
    LOG("START: " << _msg, _level); 
}

void Timer::tick() {
    auto now = Clock::now();
    auto diff = now - _last_start;
    LOG("END: " << _msg << "[" << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() / 1e6  << "s]", _level);
    _last_start = now;
}

}
