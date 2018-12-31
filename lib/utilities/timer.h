#pragma once

#include <chrono>
#include <string>
#include <sstream>
#include "utilities/log.h"

namespace cpt {

class Timer
{
public:
    using Clock = std::chrono::steady_clock;

    Timer(
        const std::string& msg,
        LogLevel level);

    void start();
    void tick();

private:
    std::string _msg;
    LogLevel _level;
    Clock::time_point _last_start;
};

#define START_TIMER_INFO(nm, msg) \
    std::stringstream _timer_ss_##nm;\
    _timer_ss_##nm << msg;\
    cpt::Timer _timer_##nm(_timer_ss_##nm.str(), cpt::LogLevel::Info);\
    _timer_##nm.start();

#define START_TIMER_DEBUG(nm, msg) \
    std::stringstream _timer_ss_##nm;\
    _timer_ss_##nm << msg;\
    cpt::Timer _timer_##nm(_timer_ss_##nm.str(), cpt::LogLevel::Debug);\
    _timer_##nm.start();

#define END_TIMER(nm) _timer_##nm.tick();

}
