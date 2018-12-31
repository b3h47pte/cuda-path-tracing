#pragma once
#include <sstream>

namespace cpt {

enum class LogLevel
{
    Debug,
    Info,
    All
};

void initialize_logging(LogLevel max_level);
void log(const std::string& msg, LogLevel level);

#define LOG(x, level) \
    {std::stringstream ss;\
    ss<<x;\
    cpt::log(ss.str(), level);}

}
