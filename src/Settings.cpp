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

#include <string>
#include <sstream>
#include <iostream>

#include "Settings.hpp"

namespace wrapd {

bool Settings::parse_args(int argc, char **argv ) {

    if (argc == 1) {
        help();
    }

    std::vector<std::string> cmd_list;
    std::vector<std::stringstream> val_list;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        // std::cout << "arg ... " << arg << std::endl;
        if ((arg.at(0) == '-') && 
                (arg.at(1) != '0') &&
                (arg.at(1) != '1') &&
                (arg.at(1) != '2') &&
                (arg.at(1) != '3') &&
                (arg.at(1) != '4') &&
                (arg.at(1) != '5') &&
                (arg.at(1) != '6') &&
                (arg.at(1) != '7') &&
                (arg.at(1) != '8') &&
                (arg.at(1) != '9') &&
                (arg.at(1) != '.')) {
            cmd_list.push_back(arg);
            val_list.push_back(std::stringstream());
        } else {
            int val_list_size = val_list.size();
            val_list[val_list_size -1] << arg << " ";
        }
    }

    const int num_commands = cmd_list.size();
    for (int i = 0; i < num_commands; i++) {
        std::string arg = cmd_list[i];
        std::stringstream val;
        val << val_list[i].rdbuf();

        if ( arg == "-help" || arg == "--help" || arg == "-h" ) {
            help();
            return true;
        } else if (arg == "-rest") { 
            std::string temp_string;
            val >> temp_string;
            m_io.rest_mesh(temp_string);
        } else if (arg == "-init") { 
            std::string temp_string;
            val >> temp_string;
            m_io.init_mesh(temp_string);
        } else if (arg == "-soln") {
            std::string temp_string;
            val >> temp_string;
            std::cout << "temp_string is: " << temp_string << std::endl;
            m_io.soln_mesh(temp_string);
        } else if (arg == "-handles") { 
            std::string temp_string;
            val >> temp_string;
            m_io.handles(temp_string);
        } else if (arg == "-p") {
            int temp_val;
            val >> temp_val;
            m_io.verbose(temp_val);
        } else if (arg == "-s") {
            bool temp_val;
            val >> temp_val;
            m_io.should_save_data(temp_val);
        //
        } else if (arg == "-it") { val >> m_max_admm_iters;
        } else if (arg == "-tol") { val >> m_tol;
        } else if (arg == "-rotaware" || arg == "-ra") {
            int temp_val;
            val >> temp_val;
            if (temp_val == 0) {
                m_rot_awareness = RotAwareness::DISABLED;
            } else if (temp_val == 1) {
                m_rot_awareness = RotAwareness::ENABLED;
            } else {
                throw std::runtime_error("Error: Invalid value provided for argument -rotaware (or -ra)");
            }
        } else if (arg == "-reweighting" || arg == "-rw") {
            int temp_val;
            val >> temp_val;
            if (temp_val == 0) {
                m_reweighting = Reweighting::DISABLED;
            } else if (temp_val == 1) {
                m_reweighting = Reweighting::ENABLED;
            } else {
                throw std::runtime_error("Error: Invalid value provided for argument -reweighting (or -rw)");
            }
        } else if (arg == "-rw_delay") { val >> m_reweighting_delay;
        } else if (arg == "-gamma") {
            double temp;
            val >> temp;
            m_gamma = temp;
        } else if (arg == "-global_it") { 
            int temp_val;
            val >> temp_val;
            m_gstep.max_iters(temp_val);
        } else if (arg == "-global_ls") {
            int temp_val;
            val >> temp_val;
            if (temp_val == 0) {
                m_gstep.linesearch(GStep::LineSearch::None);
            } else if (temp_val == 1) {
                m_gstep.linesearch(GStep::LineSearch::Backtracking);
            } else {
                throw std::runtime_error("Error: invalid value provided for argument -global_linesearch");
            }
        } else if (arg == "-kappa") { 
            double temp_val;
            val >> temp_val;
            m_gstep.kappa(temp_val);
        } else if (arg == "-model") {
            std::string temp_string;
            val >> temp_string;
            m_model_settings.elastic_model(temp_string);
        } else if (arg == "-beta_static") { 
            double temp;
            val >> temp;
            m_model_settings.elastic_beta_static(temp);
        } else if (arg == "-beta_min") {
            double temp;
            val >> temp;
            m_model_settings.elastic_beta_min(temp);
        } else if (arg == "-beta_max") {
            double temp;
            val >> temp;
            m_model_settings.elastic_beta_max(temp);
        } else if (arg == "-poisson") {
            double poisson;
            val >> poisson;
            m_model_settings.poisson(poisson);
        } else {
            throw std::runtime_error("Error: Unrecognized command-line argument: " + arg);
        }
    }

    // Check if last arg is one of our no-param args
    std::string arg(argv[argc-1]);
    if ( arg == "-help" || arg == "--help" || arg == "-h" ) {
        help();
        return true;
    }

    return false;
}  // end parse settings args

void Settings::help() {
    std::stringstream ss;
    ss << "\n==========================================\nArgs:\n" <<
        " General: \n" <<
        " -rest: zero-energy rest mesh (path to tetmesh .node file)  [REQUIRED]\n" <<
        " -init: initial deformed mesh (path to tetmesh .node file)  [REQUIRED]\n" <<
        " -handles: list of constrained vertices                     [REQUIRED]\n" <<
        " -soln: pre-computed solution (path to tetmesh .node file)  [OPTIONAL]\n" <<        
        " -p: print stats to terminal while running                  [Default: 1]       (0=disabled, 1=enabled) \n" <<
        " -s: save stats to file after running                       [Default: 1]       (0=disabled, 1=enabled) \n" <<
        " -it: maximum # ADMM iterations                             [Default: 5000]    (any integer > 0) \n" <<
        " -tol: tolerance for ADMM early exit                        [Default: 1.e-11]  (any small float > 0) \n" <<
        " -model: constitutive model for elasticity                  [Default: nh]      (nh=Neo-Hookean, arap=ARAP) \n" <<
        " -poisson: Poisson's ratio (NH model only)                  [Default: 0.45]    (any float in the range [0, 0.5)) \n" <<
        " -beta_static: weight^2 mult (no reweighting)               [Default: 1.0]     (any float >= 1.0) \n" <<
        "\n" <<
        " Rotation awareness: \n" <<
        " -ra: rotation awareness toggle                             [Default: 1]       (0=disabled, 1=enabled) \n" <<
        " -global_it: max # global step L-BFGS iters                 [Default: 100]     (any integer > 0) \n" <<        
        " -global_ls: linesearch in the global step                  [Default: 0]       (0=none, 1=backtracking) \n" <<
        " -kappa: global step early exit coeff.                      [Default: 1.0]     (any float > 0)] \n" <<
        "\n" <<
        " Dynamic reweighting: \n" <<
        " -rw: dynamic reweighting toggle                            [Default: 1]       (0=disabled, 1=enabled) \n" <<
        " -rw_delay: # ADMM iters to use static rest weights         [Default: 50]      (any integer > 0) \n" <<
        " -gamma: threshold for reweighting                          [Default:  1.5]    (any float >= 1.0) \n" <<
        " -beta_min: smallest allowable weight^2 mult                [Default:  0.1]    (any float >= 1.0) \n" <<
        " -beta_max: largest allowable weight^2 mult                 [Default: 10.0]    (any float >= 1.0) \n" <<
    "==========================================\n";
    printf("%s", ss.str().c_str() );
    exit(0);
}


}  // namespace wrapd
