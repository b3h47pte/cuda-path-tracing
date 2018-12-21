#include "json_utility.h"

#include <boost/filesystem.hpp>
#include <fstream>
#include "utilities/error.h"

namespace bfs = boost::filesystem;

namespace cpt {

nlohmann::json load_json_from_file(const std::string& filename) {
    CHECK_AND_THROW_ERROR(bfs::exists(filename), "Can not load JSON from a file that does not exist [" << filename << "].");
    std::ifstream f(filename);

    nlohmann::json jobj;
    f >> jobj;
    return jobj;
}

}
