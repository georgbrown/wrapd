// Copyright (c) 2017 University of Minnesota
//
// ADMM-Elastic Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
//    of conditions and the following disclaimer in the documentation and/or other materials
//    provided with the distribution.
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF MINNESOTA, DULUTH OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
// IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// By Matt Overby (http://www.mattoverby.net)
// Modified by George E. Brown


//
// Classes/functions used by admm-elastic samples
//

#ifndef SAMPLES_UTILS_ADDMESHES_HPP_
#define SAMPLES_UTILS_ADDMESHES_HPP_

#include <memory>
#include <numeric>
#include <vector>

#include "TetMesh.hpp"
#include "Solver.hpp"
#include "Settings.hpp"

namespace binding {

    static inline void add_tetmesh(
            wrapd::Solver *solver,
            std::shared_ptr<mcl::TetMesh> &rest_mesh,
            std::shared_ptr<mcl::TetMesh> &init_mesh,
            std::shared_ptr<mcl::TetMesh> &soln_mesh,
            const std::vector<int> &pins,
            const wrapd::Settings &settings);

    // Flags that can be added to the mesh->flags member
    enum MeshFlags {
        ARAP = 1 << 1,  // default when mesh->flags==0
        NH = 1 << 2,
    };
}  // namespace binding


//
//  Implementation
//

static inline void binding::add_tetmesh(
        wrapd::Solver *solver,
        std::shared_ptr<mcl::TetMesh> &rest_mesh,
        std::shared_ptr<mcl::TetMesh> &init_mesh,
        std::shared_ptr<mcl::TetMesh> &soln_mesh,
        const std::vector<int> &pins,
        const wrapd::Settings &settings) {
    if (rest_mesh->vertices.size() != init_mesh->vertices.size()) {
        throw std::runtime_error("Error: Rest mesh and init mesh do not have the same number of vertices.");
    }

    // Add vertices to the solver
    int num_tet_verts = rest_mesh->vertices.size();  // tet verts
    int prev_tet_verts = solver->m_system->num_all_verts();
    int num_tets = rest_mesh->tets.size();

    wrapd::math::MatX3 added_X(num_tet_verts, 3);
    wrapd::math::MatX3 added_X_init(num_tet_verts, 3);
    wrapd::math::MatX3 added_X_soln(num_tet_verts, 3);
    for (int i = 0; i < num_tet_verts; ++i) {
        int idx = i + prev_tet_verts;
        added_X.row(idx) = rest_mesh->vertices[i];
        added_X_init.row(idx) = init_mesh->vertices[i];
        added_X_soln.row(idx) = soln_mesh->vertices[i];
    }

    solver->m_system->add_nodes(added_X, added_X_init, added_X_soln);

    std::vector<int> sorted_pins(pins);
    std::sort(sorted_pins.begin(), sorted_pins.end());
    std::vector<int> free_index_list(rest_mesh->vertices.size());
    std::vector<int> fix_index_list(rest_mesh->vertices.size());

    for (int i = 0; i < static_cast<int>(fix_index_list.size()); i++) {
        fix_index_list[i] = -1;
    }

    int j = 0;
    int k = 0;
    for (int i = 0; i < static_cast<int>(free_index_list.size()); i++) {
        if (j < static_cast<int>(sorted_pins.size()) && i == sorted_pins[j]) {
            free_index_list[i] = -1;
            fix_index_list[i] = j;
            j += 1;
        } else {
            free_index_list[i] = k;
            fix_index_list[i] = -1;
            k += 1;
        }
    }

    std::shared_ptr<wrapd::TetElements>& tet_elements = solver->m_system->tet_elements();
    wrapd::create_tets_from_mesh<double, wrapd::TetElements>(
            tet_elements,
            &rest_mesh->vertices[0][0],
            &rest_mesh->tets[0][0],
            num_tets,
            settings,
            free_index_list,
            fix_index_list,
            prev_tet_verts);
    
    solver->set_pins(sorted_pins);

    if (settings.m_io.verbose() > 0) {
        std::cout << "Added mesh: " <<
            "\n\tvertices: " << num_tet_verts <<
            "\n\ttets: " << num_tets <<
            "\n\tpins: " << sorted_pins.size() <<
        std::endl;
    }
}

#endif  // SAMPLES_UTILS_ADDMESHES_HPP_
