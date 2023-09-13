#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {

    std::cout << "Performance Comparsion:" << std::endl;
    std::cout << "----------------------" << std::endl;

    int N = 10000;  // number of the subjects
    int M = 100;   // number of the variables

    srand((unsigned int)time(0));

    // Generate data where ys = ws @ xs 
    Eigen::MatrixXd xs = Eigen::MatrixXd::Random(M, N);
    Eigen::VectorXd ws = Eigen::VectorXd::Random(M);
    Eigen::VectorXd ys = ws.transpose() * xs;

    Eigen::MatrixXd x = xs;  // make copies of xs & ys before modifying them 
    Eigen::VectorXd y = ys;


    // method1
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd wy(M);
    Eigen::VectorXd sx(M);
    for (int i = 0; i < M; i++) {
        sx(i) = x.row(i).squaredNorm();
        Eigen::VectorXd projection = (x.block(i + 1, 0, M - i - 1, N) * x.row(i).transpose()) / sx(i);
        x.block(i + 1, 0, M - i - 1, N) -= projection * x.row(i);
    }

    for (int i = M - 1; i >= 0; i--) {
        wy(i) = y.dot(x.row(i)) / sx(i);
        y -= wy(i) * xs.row(i).transpose();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Method1: Solving the equation with Error Prediction & Gram-Schmidt orthogonalization" << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    std::cout << "------------------" << std::endl;
    // method2
    Eigen::MatrixXd ix = xs.transpose();
    start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd wi = ix.completeOrthogonalDecomposition().pseudoInverse() * ys;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Method2: Solving the equation using Matrix Inversion & SVD Decompostision" << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    std::cout << "------------------" << std::endl;
    std::cout << "Comparing a sample of the resulted weights from the two methods (should be similar)" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "method1: " <<  wy(i) << "  method2: " <<  wi(i) << std::endl;
    }

    return 0;
}
