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

complex<double> compute_Determinant(MatrixXcd& K) {
  FullPivLU<MatrixXcd> Kinverse;
  Kinverse.compute(K);
  complex<double> determinant;
  if (K.rows()>0) {        //      Check needed when the matrix is predominantly diagonal.
    MatrixXcd LU    =       Kinverse.matrixLU();
    determinant     =       log(LU(0,0));
    for (int k=1; k<K.rows(); ++k) {
      determinant+=log(LU(k,k));
    }
    //              Previous version which had some underflow.
    //              determinant	=	log(abs(K.determinant()));
  }
  return determinant;
};




class Alg_Kernel
    : public HODLR_Matrix {
 public:
	Alg_Kernel(const char* filename) {
    load(filename, data_);
  }

  std::complex<double> get_Matrix_Entry(const unsigned i, const unsigned j) {
    return data_(i,j);
  }

  MatrixXcd data_;
};



int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " MATRIX_DATA_FILE" << std::endl;
    std::exit(1);
  }

	// Set up the kernel.
	Alg_Kernel kernel(argv[1]);

  // Settings for the solver.
  unsigned N       = kernel.data_.rows();   // Must be symmetric?
  unsigned nLeaf   = 50;
  double tolerance = 1e-14;

  cout << N << " data points" << endl;

  // Build the RHS matrix.
	unsigned nRhs	=	1;
	MatrixXcd xExact	=	MatrixXcd::Random(N, nRhs);
	MatrixXcd bExact(N,nRhs), bFast(N,nRhs), xFast(N,nRhs);

	cout << endl << "Number of particles is: " << N << endl;
	clock_t start, end;

	cout << endl << "Setting things up..." << endl;
	start	=	clock();
	HODLR_Tree<Alg_Kernel>* A	=	new HODLR_Tree<Alg_Kernel>(&kernel, N, nLeaf);
	end		=	clock();
	cout << "Time taken is: " << double(end-start)/double(CLOCKS_PER_SEC)<< endl;

	cout << endl << "Assembling the matrix in HODLR form..." << endl;
	start			=	clock();
	VectorXcd diagonal	=	4.0*VectorXcd::Ones(N);
	A->assemble_Matrix(diagonal, tolerance);
	end		=	clock();
	cout << "Time taken is: " << double(end-start)/double(CLOCKS_PER_SEC)<< endl;

  cout << endl << "Exact matrix vector product..." << endl;
  start           =       clock();
  for (unsigned i=0; i<N; ++i) {
    bExact(i,0)             =       diagonal(i)*xExact(i,0);
    for (unsigned j=0; j<i; ++j) {
      bExact(i,0)     =       bExact(i,0)+kernel.get_Matrix_Entry(i, j)*xExact(j,0);
    }
    for (unsigned j=i+1; j<N; ++j) {
      bExact(i,0)     =       bExact(i,0)+kernel.get_Matrix_Entry(i, j)*xExact(j,0);
    }
  }
  end		=	clock();
	cout << "Time taken is: " << double(end-start)/double(CLOCKS_PER_SEC)<< endl;

	cout << endl << "Fast matrix matrix product..." << endl;
	start		=	clock();
	A->matMatProduct(xExact, bFast);
	end		=	clock();
	cout << "Time taken is: " << double(end-start)/double(CLOCKS_PER_SEC)<< endl;

	cout << endl << "Factoring the matrix..." << endl;
	start		=	clock();
	A->compute_Factor();
	end		=	clock();
	cout << "Time taken is: " << double(end-start)/double(CLOCKS_PER_SEC)<< endl;

	cout << endl << "Solving the system..." << endl;
	start		=	clock();
	A->solve(bExact, xFast);
	end		=	clock();
	cout << "Time taken is: " << double(end-start)/double(CLOCKS_PER_SEC)<< endl;

	cout << endl << "Error in computed solution: " << (xFast-xExact).norm()/xExact.norm()<< endl;

	cout << endl << "Error in matrix matrix product: " << (bFast-bExact).cwiseAbs().maxCoeff() << endl;

	complex<double> determinant;
	cout << endl << "Computing the log determinant..." << endl;
	start		=	clock();
	A->compute_Determinant(determinant);
	end		=	clock();
	cout << "Time taken is: " << double(end-start)/double(CLOCKS_PER_SEC)<< endl;

	cout << endl << "Log determinant is: " << std::setprecision(16) << determinant << endl;


  MatrixXcd K;
  kernel.get_Matrix(0, 0, N, N, K);
  for (unsigned k=0; k<N; ++k) {
    K(k,k)  =       diagonal(k);
  }

  complex<double> exact_determinant;
  exact_determinant       =       compute_Determinant(K);
  cout << endl << "Exact log determinant is: " << std::setprecision(16) << exact_determinant << endl;

  return 0;
}
