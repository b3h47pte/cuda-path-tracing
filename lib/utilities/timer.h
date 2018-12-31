#pragma once

#include <chrono>
#include <string>
#include <sstream>

namespace cpt {

class Timer
{
public:
    using Clock = std::chrono::steady_clock;

    Timer(const std::string& msg);

    void start();
    void tick();

private:
    std::string _msg;
    Clock::time_point _last_start;
};

#define START_TIMER(nm, msg) \
    std::stringstream _timer_ss_##nm;\
    _timer_ss_##nm << msg;\
    cpt::Timer _timer_##nm(_timer_ss_##nm.str());\
    _timer_##nm.start();

#define END_TIMER(nm) _timer_##nm.tick();

}
