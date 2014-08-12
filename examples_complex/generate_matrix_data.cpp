#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <complex>

#include <Eigen/Dense>

#include "HODLR_Tree.hpp"
#include "HODLR_Matrix.hpp"
#include "KDTree.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::vector;
using std::getline;
using Eigen::VectorXcd;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using namespace Eigen;

/** Input an Eigen matrix from filename in row-major */
template <typename MatrixType>
void load(const char* filename, MatrixType& m) {
  std::ifstream file(filename);
  if (file.is_open()) {
    typename MatrixType::Index rows, cols;
    file >> rows;
    file >> cols;
    m = MatrixType(rows, cols);
    for (unsigned i = 0; i < m.rows(); ++i)
      for (unsigned j = 0; j < m.cols(); ++j)
        file >> m(i,j);
  } else {
    std::cerr << "Error reading file" << std::endl;
    std::exit(1);
  }
}

/** Output an Eigen matrix to filename in row-major */
template <typename MatrixType>
void save(const char* filename, const MatrixType& m) {
  std::ofstream file(filename);
  if (file.is_open()) {
    file << m.rows() << "\n";
    file << m.cols() << "\n";
    for (unsigned i = 0; i < m.rows(); ++i)
      for (unsigned j = 0; j < m.cols(); ++j)
        file << m(i,j) << " ";
  } else {
    std::cerr << "Error writing file" << std::endl;
    std::exit(1);
  }
}



int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " MATRIX_DATA_FILE" << std::endl;
    std::exit(1);
  }

  // Write some test data by evaluating a kernel

  // Generate random points
  const unsigned N = 2000;
  const unsigned nDim = 3;
  MatrixXd points = MatrixXd::Random(N, nDim);
  get_KDTree_Sorted(points,0);   // cheating

  // Fill the matrix
  MatrixXcd kernel_matrix(N,N);
  for (unsigned i = 0; i < kernel_matrix.rows(); ++i) {
    for (unsigned j = 0; j < kernel_matrix.cols(); ++j) {
      double R2	=	(points(i,0)-points(j,0))*(points(i,0)-points(j,0));
      for (unsigned k=1; k<nDim; ++k) {
        R2 += (points(i,k)-points(j,k))*(points(i,k)-points(j,k));
      }
      kernel_matrix(i,j) = std::complex<double>(sqrt(1.0+R2), 2.0);
    }
  }

  // Save to file
  save(argv[1], kernel_matrix);
}
