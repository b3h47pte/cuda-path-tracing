#include "log.h"
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include "utilities/error.h"

namespace cpt {
namespace {

boost::log::trivial::severity_level log_level_to_boost(LogLevel lvl) {
    switch(lvl) {
    case LogLevel::Debug:
        return boost::log::trivial::debug;
    case LogLevel::Info:
        return boost::log::trivial::info;
        break;
    default:
        break;
    }
    return boost::log::trivial::fatal;
}

}

void initialize_logging(LogLevel max_level) {
    boost::log::core::get()->set_filter(
        boost::log::trivial::severity <= log_level_to_boost(max_level)
    );
}

void log(const std::string& msg, LogLevel level) {
    switch(level) {
    case LogLevel::Debug:
        BOOST_LOG_TRIVIAL(debug) << msg;
        break;
    case LogLevel::Info:
        BOOST_LOG_TRIVIAL(info) << msg;
        break;
    default:
        THROW_ERROR("Unsupported log type.");
    }
}

}
