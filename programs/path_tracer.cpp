#include <boost/program_options.hpp>
#include <gpgpu/cuda_utils.h>
#include <iostream>
#include <render/cuda_renderer.h>
#include <scene/loader/scene_loader.h>
#include <string>
#include <utilities/filesystem_utility.h>
#include <utilities/json_utility.h>
#include <utilities/log.h>
#include <utilities/timer.h>

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
    cpt::initialize_logging(cpt::LogLevel::Debug);
    cpt::initialize_cuda();

    // Load scene.
    const std::string sceneFname = vm["scene"].as<std::string>();
    START_TIMER_INFO(scene_loader, "Loading Scene [" << sceneFname << "]...");
    cpt::SceneLoader loader;
    cpt::ScenePtr scene = loader.load_scene_from_json(
        cpt::load_json_from_file(sceneFname),
        cpt::get_parent_directory(sceneFname));
    END_TIMER(scene_loader);

    // Render.
    cpt::AovOutput output;

    // TODO: Pull from options.
    output.initialize(640, 480);
    const std::string render_camera = "main";

    START_TIMER_INFO(create_renderer, "Create renderer...");
    cpt::CudaRenderer rndr(scene, render_camera);
    END_TIMER(create_renderer);

    START_TIMER_INFO(render, "Render...");
    rndr.render(output);
    END_TIMER(render);

    // Save image.
    // TODO: Make file path pull from options.
    // TODO: Tonemapping?.
    START_TIMER_INFO(save, "Saving...");
    output.save_channel_to_file(cpt::AovOutput::Channels::FinalImage, "tmp.tiff");
    END_TIMER(save);

    return 0;
}
