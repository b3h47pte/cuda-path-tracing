#include <boost/program_options.hpp>
#include "gpgpu/cuda_utils.h"
#include <iostream>
#include "scene/loader/scene_loader.h"
#include <string>

namespace po = boost::program_options;

int main(int argc, char** argv) {
    // Parse command line.
    po::options_description desc("Options");
    desc.add_options()
        ("scene", po::value<std::string>()->required(), "JSON scene file.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Initialize library.
    cpt::initialize_cuda();

    // Load scene.
    const std::string sceneFname = vm["scene"].as<std::string>();
    std::cout << "Loading Scene [" << sceneFname << "]..." << std::endl;
    cpt::ScenePtr scene = cpt::loadSceneFromJson(sceneFname);

    // Render.

    // Save image.
    return 0;
}
