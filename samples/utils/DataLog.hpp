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

#ifndef SAMPLES_UTILS_DATALOG_HPP_
#define SAMPLES_UTILS_DATALOG_HPP_

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include "Math.hpp"

class DataLog {
 public:
    explicit DataLog(std::string name);

    void write(std::string outdir, std::string prefix, std::string ext);
    void addPoint(double x, double y);

    std::vector<Eigen::Vector2d> points;
    std::string m_name;

 private:
};


//
// Implementation
//

DataLog::DataLog(std::string name)
    : m_name(name) {
}

void DataLog::write(std::string outdir, std::string prefix, std::string ext) {
    std::stringstream fileout;
    fileout << outdir << prefix << m_name << ext;
    std::ofstream outstream(fileout.str().c_str());
    if (outstream.is_open()) {
        for (int i = 0; i < static_cast<int>(points.size()); i++) {
            double xval = points[i][0];
            double yval = points[i][1];
            outstream << std::scientific << std::setprecision(14) << xval << " "
                    << std::scientific << std::setprecision(14) << yval << std::endl;
        }
    } else {
        throw std::runtime_error("Error: Unable to open file.");
    }
    outstream.close();
}

void DataLog::addPoint(double x, double y) {
    points.push_back(Eigen::Vector2d(x, y));
}


#endif  // SAMPLES_UTILS_DATALOG_HPP_
