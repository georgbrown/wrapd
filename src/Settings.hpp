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

#ifndef SRC_SETTINGS_HPP_
#define SRC_SETTINGS_HPP_

#include <filesystem>
#include <string>

#include "Math.hpp"
#include "ModelSettings.hpp"

namespace wrapd {
// Solver settings
struct Settings {
    bool parse_args(int argc, char **argv);  // parse from terminal args. Returns true if help()
    void help();  // -help  print details, parse_args returns true if used

    //
    // IO Settings
    //
    class IO {
     public:
        IO() :
            m_rest_mesh(""),
            m_init_mesh(""),
            m_soln_mesh(""),
            m_handles(""),
            m_verbose(1),
            m_soln_provided(false) {
        }

        std::string rest_mesh() const { return m_rest_mesh; }
        void rest_mesh(std::string val) { 
            m_rest_mesh = strip_extension(val);
        }

        std::string init_mesh() const { return m_init_mesh; }
        void init_mesh(std::string val) { 
            m_init_mesh = strip_extension(val);
        }

        std::string soln_mesh() const { return m_soln_mesh; }
        void soln_mesh(std::string val) { 
            m_soln_mesh = strip_extension(val);
            m_soln_provided = true; 
        }

        std::string handles() const { return m_handles; }
        void handles(std::string val) { m_handles = val; }

        int verbose() const { return m_verbose; }
        void verbose(int val) { m_verbose = val; }

        bool should_save_data() const { return m_should_save_data; }
        void should_save_data(bool val) { m_should_save_data = val; }

        bool soln_provided() const { return m_soln_provided; }

     private:

        std::string strip_extension(std::string raw_input) {
            std::string parent_path = std::string(std::filesystem::path(raw_input).parent_path());
            std::string stem = std::string(std::filesystem::path(raw_input).stem());
            std::string path;
            if (parent_path == "") {
                path = stem;
            } else {
                path = parent_path + "/" + stem;
            }
            return path; 
        }

        std::string m_rest_mesh;
        std::string m_init_mesh;
        std::string m_soln_mesh;
        std::string m_handles;
        int m_verbose;
        bool m_should_save_data;
        bool m_soln_provided;
    };
    IO m_io;

    double m_tol;  // -tol <double>  obj. resid. tolerance for early exit (unless a solution is provided for X error comparisons)
    int m_max_admm_iters;  // -it <int>  number of admm solver iterations
    double m_reweighting_delay;

    enum class RotAwareness {
        ENABLED,
        DISABLED
    };
    RotAwareness m_rot_awareness;

    enum class Reweighting {
        DISABLED,
        ENABLED
    };
    Reweighting m_reweighting;

    double m_gamma;

    class GStep {
     public:
        GStep() :
            m_max_iters(100),
            m_linesearch(LineSearch::None),
            m_kappa(1.0) {
        }

        enum class LineSearch {
            None,
            Backtracking,
        };

        int max_iters() const { return m_max_iters; }
        void max_iters(int val) { m_max_iters = val; }

        LineSearch linesearch() const { return m_linesearch; }
        void linesearch(LineSearch val) { m_linesearch = val; }

        double kappa() const { return m_kappa; }
        void kappa(double val) { m_kappa = val; }

     private:

        int m_max_iters;
        LineSearch m_linesearch;
        double m_kappa;
    };
    GStep m_gstep;

    ModelSettings m_model_settings;

    //
    Settings() {            
            m_rot_awareness = RotAwareness::ENABLED;
            m_reweighting = Reweighting::ENABLED;
            m_gamma = 1.5;
            m_tol = 1.e-11;
            m_max_admm_iters = 5000;
            m_reweighting_delay = 50;

    }
};
}  // namespace wrapd
#endif  // SRC_SETTINGS_HPP_
