#include <iostream>
#include <Eigen/Dense>
#include <chrono>

//method1 = our prediction error method.
//method2 = matrix inversion method.

int main() {
    int N = 1000;  // number of the subjects
    int M = 100;   // number of the variables

    // Seed random number generator
    srand((unsigned int)time(0));

    // Generate random matrices using Eigen
    Eigen::MatrixXd xs = Eigen::MatrixXd::Random(M, N);
    Eigen::VectorXd ws = Eigen::VectorXd::Random(M);
    Eigen::VectorXd ys = ws.transpose() * xs;

    Eigen::MatrixXd x = xs;  // make copies of xs & ys before modifying them 
    Eigen::VectorXd y = ys;

    auto start = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd wy = Eigen::VectorXd::Zero(M);
    Eigen::VectorXd sx = Eigen::VectorXd::Zero(M);

    for (int i = 0; i < M; i++) {
        sx(i) = x.row(i).squaredNorm();
        for (int j = i + 1; j < M; j++) {
            double wx = (x.row(i) * x.row(j).transpose()).sum() / sx(i);
            x.row(j) -= wx * x.row(i);
        }
    }

    for (int i = M - 1; i >= 0; i--) {
        wy(i) = y.dot(x.row(i)) / sx(i);
        y -= wy(i) * xs.row(i).transpose();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time method1: " << elapsed.count() << " seconds" << std::endl;

    Eigen::MatrixXd ix = xs.transpose();
    start = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd wi = ix.completeOrthogonalDecomposition().pseudoInverse() * ys;

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time method2: " << elapsed.count() << " seconds" << std::endl;

    std::cout << "--------------" << std::endl;
    std::cout << "comparing the resulted weights from the two methods" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "method1: " <<  wy(i) << " - method2: " <<  wi(i) << std::endl;
    }

    return 0;
}
