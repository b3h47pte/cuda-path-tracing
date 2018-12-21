#include "filesystem_utility.h"
#include <boost/filesystem.hpp>

namespace bfs = boost::filesystem;

namespace cpt {

std::string get_parent_directory(const std::string& dir) {
    const bfs::path path(dir);
    return path.parent_path().native();
}

}
