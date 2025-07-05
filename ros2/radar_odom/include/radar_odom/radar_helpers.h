#ifndef __RADAR_HELPERS_H__
#define __RADAR_HELPERS_H__

// get an estimate of how good your least squares fit was
std::tuple<float, float> r_squared(const Eigen::MatrixXf& DX, const Eigen::VectorXf& DY, const Eigen::Vector2f& model)
{
    // Compute average
    const float y_avg = DY.mean();

    // Compute total sum of squares
    const int N = DX.rows();
    // if (N>2) {
    //     std::cout << "y: \n" << DY << std::endl;
    // }
    const float sum_squares = (DY - (y_avg * Eigen::VectorXf::Ones(N))).squaredNorm();
    // std::cout << "diff: \n" << (DY - (y_avg * Eigen::VectorXf::Ones(N))) << std::endl;
    // std::cout << "squares: \n" << sum_squares << std::endl;

    // Compute predicted values
    const Eigen::VectorXf estimated_DY = DX * model;

    // Compute sum of residual squared
    const float sum_residuals_square = (DY - estimated_DY).squaredNorm();

    // Computer coefficient of determination
    float R = 1 - (sum_residuals_square / sum_squares);

    return std::make_tuple(R, sum_residuals_square);

}

// returns transpose of 2d matrix
std::vector<std::vector<double>> getTranspose(const std::vector<std::vector<double>> matrix1) {
    //Transpose-matrix: height = width(matrix), width = height(matrix)
    std::vector<std::vector<double>> solution(matrix1[0].size(), std::vector<double> (matrix1.size()));

    //Filling solution-matrix
    for(size_t i = 0; i < matrix1.size(); i++) {
        for(size_t j = 0; j < matrix1[0].size(); j++) {
            solution[j][i] = matrix1[i][j];
        }
    }
    return solution;
}

// performs matrix multiplication
std::vector<std::vector<double>> matMult(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B) {
    int rowsA =  A.size();
    int colsA = A[0].size();

    int colsB = B[0].size();

    std::vector<std::vector<double>> solution;

    // initialize solution matrix with 0
    for(int i = 0; i < rowsA; i++) {
        solution.push_back(std::vector<double>());
        for(int j = 0; j < colsB; j++) {
            solution[i].push_back(0);
        }
    }

    // perform the multiplication
    for(int i = 0; i < rowsA; i++) {
        for(int j = 0; j < colsB; j++) {
            for(int k = 0; k < colsA; ++k) {
                solution[i][j] += A[i][k] * B[k][j];
            }    
        }
    }

    return solution;
}

// performs matrix-vector multiplication
std::vector<double> matvecMult(std::vector<std::vector<double>> A, std::vector<double> b)
{
    int rowsA =  A.size();

    int lenb =  b.size();  

    std::vector<double> solution;

    // initialize solution matrix with 0
    for(int i = 0; i < rowsA; i++) {
        for(int j = 0; j < lenb; j++) {
            solution.push_back(0);
        }
    }

    // perform the multiplication
    for(int i = 0; i < rowsA; i++) {
        for(int j = 0; j < lenb; j++) {
            solution[i] += A[i][j] * b[j];    
        }
    }

    return solution;
}

#endif