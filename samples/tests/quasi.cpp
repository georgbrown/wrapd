// Copyright (c) 2021 George E. Brown and Rahul Narain
//
// WRAPD uses the MIT License (https://opensource.org/licenses/MIT)
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is furnished
// to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// By George E. Brown (https://www-users.cse.umn.edu/~brow2327/)

#include "AddMeshes.hpp"
#include "MeshIO.hpp"
#include "DataLog.hpp"

std::vector<int> pins;

DataLog objectives_log("objectives");
DataLog x_errors_log("x_errors");
DataLog reweighted_log("reweighted");
DataLog accumulated_time_s_log("accumulated_time_s");

std::vector<double> objectives;
std::vector<double> x_errors;
std::vector<int> reweighted;
std::vector<double> accumulated_time_s;
wrapd::math::MatX3 final_positions;
std::shared_ptr<mcl::TetMesh> final_mesh_elenode;

void load_handles(std::string handles_file);
void post_solve(std::shared_ptr<wrapd::Solver> solver);

int main(int argc, char **argv) {
    wrapd::Settings settings;
    settings.parse_args(argc, argv);

    // Create the init and rest meshes
    std::stringstream handles_file;
    std::stringstream rest_file;
    std::stringstream init_file;
    std::stringstream soln_file;

    handles_file << settings.m_io.handles();
    rest_file << settings.m_io.rest_mesh();
    init_file << settings.m_io.init_mesh();
    if (settings.m_io.soln_provided()) {
        soln_file << settings.m_io.soln_mesh();
    }

    load_handles(handles_file.str());

    std::shared_ptr<wrapd::Solver> solver = std::make_shared<wrapd::Solver>(settings);

    std::shared_ptr<mcl::TetMesh> rest_mesh;
    std::shared_ptr<mcl::TetMesh> init_mesh;
    rest_mesh = mcl::TetMesh::create();
    init_mesh = mcl::TetMesh::create();
    mcl::meshio::load_elenode(rest_mesh.get(), rest_file.str());
    mcl::meshio::load_elenode(init_mesh.get(), init_file.str());

    std::shared_ptr<mcl::TetMesh> soln_mesh;
    if (!settings.m_io.soln_mesh().empty()) {
        soln_mesh = mcl::TetMesh::create();
        mcl::meshio::load_elenode(soln_mesh.get(), soln_file.str());
    } else {
        soln_mesh = rest_mesh;
    }

    final_mesh_elenode = mcl::TetMesh::create();
    final_mesh_elenode->tets = init_mesh->tets;
    final_mesh_elenode->vertices.resize(init_mesh->vertices.size());

    // Setting flags
    if (settings.m_model_settings.elastic_model() == wrapd::ModelSettings::ElasticModel::ARAP) {
        rest_mesh->flags |= binding::ARAP;
    } else if (settings.m_model_settings.elastic_model() == wrapd::ModelSettings::ElasticModel::NH) {
        rest_mesh->flags |= binding::NH;
    } else {
        throw std::runtime_error("Error: Invalid elastic");
    }
    init_mesh->flags = rest_mesh->flags;
    binding::add_tetmesh(solver.get(), rest_mesh, init_mesh, soln_mesh, pins, settings);

    solver->initialize(settings);
    solver->solve();
    post_solve(solver);

    // Write data
    std::stringstream outdir_ss;
    outdir_ss << WRAPD_OUTPUT_DIR << "/";
    bool writeData = settings.m_io.should_save_data();
    if (writeData) {
        for (size_t i = 0; i < objectives.size(); i++) {
            objectives_log.addPoint(i, objectives[i]);
        }
        for (size_t i = 0; i < x_errors.size(); i++) {
            x_errors_log.addPoint(i, x_errors[i]);
        }
        for (size_t i = 0; i < reweighted.size(); i++) {
            reweighted_log.addPoint(i, reweighted[i]);
        }
        for (size_t i = 0; i < accumulated_time_s.size(); i++) {
            accumulated_time_s_log.addPoint(i, accumulated_time_s[i]);
        }
        
        std::string prefix = "";
        std::string extension = ".txt";
        objectives_log.write(outdir_ss.str(), prefix, extension);
        x_errors_log.write(outdir_ss.str(), prefix, extension);
        reweighted_log.write(outdir_ss.str(), prefix, extension);
        accumulated_time_s_log.write(outdir_ss.str(), prefix, extension);
    }

    // Saving the final tet mesh
    std::stringstream final_mesh_elenode_file;
    final_mesh_elenode_file << WRAPD_OUTPUT_DIR << "/" << "final_mesh";
    const wrapd::math::MatX3 &X = solver->m_system->X();
    for (int i = 0; i < static_cast<int>(final_mesh_elenode->vertices.size()); i++) {
        final_mesh_elenode->vertices[i] = X.row(i);
    }
    mcl::meshio::save_elenode(final_mesh_elenode.get(), final_mesh_elenode_file.str());
    
    // Saving the final mesh as an .obj (has no tets info, only verts and faces)
    final_mesh_elenode->need_faces(true);
    std::shared_ptr<mcl::TriangleMesh> final_mesh_obj = mcl::TriangleMesh::create();
    final_mesh_obj->vertices = final_mesh_elenode->vertices;
    final_mesh_obj->faces = final_mesh_elenode->faces;
    std::stringstream final_mesh_obj_file;
    final_mesh_obj_file << WRAPD_OUTPUT_DIR << "/" << "final_mesh.obj";
    mcl::meshio::save_obj(final_mesh_obj.get(), final_mesh_obj_file.str());

    return EXIT_SUCCESS;
}

void load_handles(std::string handles_file) {
    pins.clear();
    std::ifstream infile(handles_file.c_str());
    if (infile.is_open()) {
        std::string line;
        while (std::getline(infile, line)) {
            std::stringstream ss(line);
            int index;
            while (ss >> index) {
                pins.emplace_back(index);
            }
        }
    }
}

void post_solve(std::shared_ptr<wrapd::Solver> solver) {
    wrapd::AlgorithmData* algorithm_data = solver->algorithm_data();

    objectives = algorithm_data->per_iter_objectives();
    x_errors = algorithm_data->per_iter_state_x_errors();
    accumulated_time_s = algorithm_data->per_iter_accumulated_time_s();
    reweighted = algorithm_data->per_iter_reweighted();

    std::cout << std::fixed;
    double local_s = algorithm_data->m_runtime.local_ms / 1000.;
    double global_s = algorithm_data->m_runtime.global_ms / 1000.;
    double refactor_s = algorithm_data->m_runtime.refactor_ms / 1000.;

    if (solver->m_settings.m_io.verbose() > 0) {
        std::cout << "Final accumulated time (s): " << (local_s + global_s + refactor_s) << std::endl;
        std::cout << "-- Local step (s): " << local_s << std::endl;
        std::cout << "-- Global step (s): " << global_s << std::endl;
        std::cout << "-- Refactor step (s): " << refactor_s << std::endl;
    }
}