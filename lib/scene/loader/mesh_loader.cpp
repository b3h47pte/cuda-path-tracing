#include "mesh_loader.h"

#include <algorithm>
#include <boost/filesystem.hpp>
#include "scene/geometry/geometry_aggregate.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"
#include "utilities/error.h"
#include <vector>

namespace bfs = boost::filesystem;

namespace cpt {
namespace {

const std::string OBJ_EXTENSION = ".obj";

}

GeometryPtr load_mesh_from_file(const std::string& fname) {
    bfs::path fpath(fname);    

    // Dispatch based on file extension.
    std::string extension = fpath.extension().native();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    if (extension == OBJ_EXTENSION) {
        return load_obj_from_file(fname);
    } else {
        THROW_ERROR("Mesh extension type [" << extension << "] is not supported.");
    }

    return nullptr;
}

GeometryPtr load_obj_from_file(const std::string& fname) {
    GeometryAggregateBuilder builder;

    const std::string base_path = bfs::path(fname).parent_path().native() + "/";

    tinyobj::attrib_t obj_attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    bool ok = tinyobj::LoadObj(&obj_attrib, &shapes, &materials, &err, fname.c_str(), base_path.c_str());
    CHECK_AND_THROW_ERROR(ok, "Failed to load OBJ File [" << fname << "]." << std::endl << err);

    // Go through all the vertex data and store them in the GeometryAggregate.
    {
        CHECK_AND_THROW_ERROR((obj_attrib.vertices.size() % 3) == 0, "Vertex positions not 3D [" << fname << "].");
        const size_t total_vertices = obj_attrib.vertices.size() / 3;
        for (size_t i = 0; i < total_vertices; ++i) {
            Eigen::Vector3f position(obj_attrib.vertices[i * 3], obj_attrib.vertices[i * 3 + 1], obj_attrib.vertices[i * 3 + 2]);
            builder.add_vertex_position(position);
        }
    }

    {
        CHECK_AND_THROW_ERROR((obj_attrib.normals.size() % 2) == 0, "Vertex UV not 2D [" << fname << "].");
        const size_t total_vertices = obj_attrib.texcoords.size() / 2;
        for (size_t i = 0; i < total_vertices; ++i) {
            Eigen::Vector2f uv(obj_attrib.texcoords[i * 2], obj_attrib.texcoords[i * 2 + 1]);
            builder.add_vertex_uv(uv);
        }
    }

    {
        CHECK_AND_THROW_ERROR((obj_attrib.normals.size() % 3) == 0, "Vertex normals not 3D [" << fname << "].");
        const size_t total_vertices = obj_attrib.normals.size() / 3;
        for (size_t i = 0; i < total_vertices; ++i) {
            Eigen::Vector3f normal(obj_attrib.normals[i * 3], obj_attrib.normals[i * 3 + 1], obj_attrib.normals[i * 3 + 2]);
            builder.add_vertex_normal(normal);
        }
    }


    // Iterate through every shape and combine all the faces into a single cpt::GeometryAggregate object.
    for (size_t i = 0; i < shapes.size(); ++i) {
        size_t idx_counter = 0;
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); ++f) {
            const int num_vertices = shapes[i].mesh.num_face_vertices[f];
            CHECK_AND_THROW_ERROR(num_vertices == 3, "OBJ not made up of triangles [" << fname << "].");

            tinyobj::index_t indices[3];
            for (auto j = 0; j < 3; ++j)
                indices[j] = shapes[i].mesh.indices[idx_counter + j];

            builder.add_face(
                Eigen::Vector3i(indices[0].vertex_index, indices[1].vertex_index, indices[2].vertex_index),
                Eigen::Vector3i(indices[0].texcoord_index, indices[1].texcoord_index, indices[2].texcoord_index),
                Eigen::Vector3i(indices[0].normal_index, indices[1].normal_index, indices[2].normal_index));

            idx_counter += num_vertices;
        }
    }

    return builder.construct();
}

}
