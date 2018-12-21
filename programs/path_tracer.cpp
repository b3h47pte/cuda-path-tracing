#include <boost/program_options.hpp>
#include <gpgpu/cuda_utils.h>
#include <iostream>
#include <scene/loader/scene_loader.h>
#include <string>
#include <utilities/filesystem_utility.h>
#include <utilities/json_utility.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
    // Parse command line.
    po::options_description desc("Options");
    desc.add_options()
        ("scene", po::value<std::string>()->required(), "JSON scene file.")
        ("options", po::value<std::string>()->required(), "JSON options file.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Initialize library.
    cpt::initialize_cuda();

    // Load scene.
    const std::string sceneFname = vm["scene"].as<std::string>();
    std::cout << "Loading Scene [" << sceneFname << "]..." << std::endl;
    cpt::ScenePtr scene = cpt::load_scene_from_json(
        cpt::load_json_from_file(sceneFname),
        cpt::get_parent_directory(sceneFname));

    // Render.

    // Save image.
    return 0;
}
