#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "libbsmp/block_sparse_matrix.h"
#include "libbsmp/generalized_eigen.h"
#include "libbsmp/triplet_loader.h"

int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(6);

    // Параметры
    std::string fileA = (argc > 1) ? argv[1] : "data/matrix_A_simple.txt";
    std::string fileB = (argc > 2) ? argv[2] : "data/matrix_B_simple.txt";
    int block_size = (argc > 3) ? std::atoi(argv[3]) : 4;
    int num_eigenvalues = (argc > 4) ? std::atoi(argv[4]) : 6;

    std::cout << "=== Тест обобщенной задачи на собственные значения ===" << std::endl;
    std::cout << "Матрица A: " << fileA << std::endl;
    std::cout << "Матрица B: " << fileB << std::endl;
    std::cout << "Размер блока: " << block_size << std::endl;
    std::cout << "Число собственных значений: " << num_eigenvalues << std::endl;
    std::cout << std::endl;

    // Загрузка матрицы A
    BlockSparseMatrixConfig configA;
    std::vector<int> block_rows_A, block_cols_A;
    std::vector<float> block_data_A;

    if (!load_triplet_file_as_block_sparse(fileA, block_size, configA,
                                           block_rows_A, block_cols_A, block_data_A)) {
        std::cerr << "Ошибка загрузки матрицы A из " << fileA << std::endl;
        return 1;
    }

    std::cout << "Матрица A загружена:" << std::endl;
    std::cout << "  Размерность: " << configA.num_rows << " x " << configA.num_cols << std::endl;
    std::cout << "  Ненулевых блоков: " << configA.num_nonzero_blocks << std::endl;
    std::cout << std::endl;

    // Загрузка матрицы B
    BlockSparseMatrixConfig configB;
    std::vector<int> block_rows_B, block_cols_B;
    std::vector<float> block_data_B;

    if (!load_triplet_file_as_block_sparse(fileB, block_size, configB,
                                           block_rows_B, block_cols_B, block_data_B)) {
        std::cerr << "Ошибка загрузки матрицы B из " << fileB << std::endl;
        return 1;
    }

    std::cout << "Матрица B загружена:" << std::endl;
    std::cout << "  Размерность: " << configB.num_rows << " x " << configB.num_cols << std::endl;
    std::cout << "  Ненулевых блоков: " << configB.num_nonzero_blocks << std::endl;
    std::cout << std::endl;

    // Проверка совместимости
    if (configA.num_rows != configB.num_rows || configA.num_cols != configB.num_cols) {
        std::cerr << "Ошибка: матрицы A и B имеют разные размерности!" << std::endl;
        return 1;
    }

    // Создание матриц
    BlockSparseMatrix A(configA);
    A.initialize(block_rows_A, block_cols_A, block_data_A);

    BlockSparseMatrix B(configB);
    B.initialize(block_rows_B, block_cols_B, block_data_B);

    // Решение обобщенной задачи на собственные значения
    std::cout << "========================================" << std::endl;
    std::cout << "Вычисление собственных значений..." << std::endl;
    std::cout << "========================================" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    bsmp::EigenResult result = bsmp::solveGeneralizedEigen(A, B, num_eigenvalues, 100, 1e-5f);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "========================================" << std::endl;
    std::cout << "Время вычисления: " << duration.count() / 1000.0 << " секунд" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Вывод результатов
    if (result.converged) {
        std::cout << "✓ Решение сошлось!" << std::endl;
    } else {
        std::cout << "✗ Решение не полностью сошлось" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Найденные собственные значения:" << std::endl;
    for (size_t i = 0; i < result.eigenvalues.size(); ++i) {
        std::cout << "  λ[" << i << "] = " << result.eigenvalues[i] << std::endl;
    }
    std::cout << std::endl;

    // Вывод первых нескольких компонент первого собственного вектора
    if (!result.eigenvectors.empty() && !result.eigenvectors[0].empty()) {
        std::cout << "Первый собственный вектор (первые 10 компонент):" << std::endl;
        int n_show = std::min(10, (int)result.eigenvectors[0].size());
        for (int i = 0; i < n_show; ++i) {
            std::cout << "  v[" << i << "] = " << result.eigenvectors[0][i] << std::endl;
        }
        if ((int)result.eigenvectors[0].size() > n_show) {
            std::cout << "  ..." << std::endl;
        }
    }
    std::cout << std::endl;

    // Проверка качества решения для первого собственного значения
    if (!result.eigenvalues.empty() && !result.eigenvectors.empty()) {
        std::cout << "Проверка качества (A*v vs λ*B*v для первого собственного значения):" << std::endl;

        int n = configA.num_rows;
        float lambda = result.eigenvalues[0];
        std::vector<float>& v = result.eigenvectors[0];

        // Выделяем память на устройстве
        float *d_v, *d_Av, *d_Bv;
        cudaMalloc(&d_v, n * sizeof(float));
        cudaMalloc(&d_Av, n * sizeof(float));
        cudaMalloc(&d_Bv, n * sizeof(float));

        // Копируем вектор на устройство
        cudaMemcpy(d_v, v.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        // Вычисляем A*v и B*v
        A.multiply(d_v, d_Av);
        B.multiply(d_v, d_Bv);

        // Копируем результаты обратно
        std::vector<float> Av(n), Bv(n);
        cudaMemcpy(Av.data(), d_Av, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Bv.data(), d_Bv, n * sizeof(float), cudaMemcpyDeviceToHost);

        // Вычисляем норму разности A*v - λ*B*v
        float norm_diff = 0.0f;
        float norm_Av = 0.0f;
        for (int i = 0; i < n; ++i) {
            float diff = Av[i] - lambda * Bv[i];
            norm_diff += diff * diff;
            norm_Av += Av[i] * Av[i];
        }
        norm_diff = std::sqrt(norm_diff);
        norm_Av = std::sqrt(norm_Av);

        float relative_error = norm_diff / norm_Av;

        std::cout << "  ||A*v - λ*B*v|| = " << norm_diff << std::endl;
        std::cout << "  ||A*v|| = " << norm_Av << std::endl;
        std::cout << "  Относительная ошибка = " << relative_error << std::endl;

        if (relative_error < 1e-3f) {
            std::cout << "  ✓ Качество хорошее!" << std::endl;
        } else if (relative_error < 1e-2f) {
            std::cout << "  ~ Качество приемлемое" << std::endl;
        } else {
            std::cout << "  ✗ Качество неудовлетворительное" << std::endl;
        }

        cudaFree(d_v);
        cudaFree(d_Av);
        cudaFree(d_Bv);
    }

    // Сохраняем результаты для сравнения с MATLAB
    std::ofstream eigenvals_file("data/eigen_cpp_eigenvalues.txt");
    if (eigenvals_file) {
        for (size_t i = 0; i < result.eigenvalues.size(); i++) {
            eigenvals_file << result.eigenvalues[i] << "\n";
        }
        eigenvals_file.close();
        std::cout << "\nСобственные значения сохранены в data/eigen_cpp_eigenvalues.txt" << std::endl;
    }

    // Сохраняем первый собственный вектор
    if (!result.eigenvectors.empty()) {
        std::ofstream eigenvec_file("data/eigen_cpp_eigenvector1.txt");
        if (eigenvec_file) {
            for (size_t i = 0; i < result.eigenvectors[0].size(); i++) {
                eigenvec_file << result.eigenvectors[0][i] << "\n";
            }
            eigenvec_file.close();
            std::cout << "Первый собственный вектор сохранён в data/eigen_cpp_eigenvector1.txt" << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "=== Тест завершен ===" << std::endl;
    std::cout << "Сравните с MATLAB: cd generators && matlab -r test_eigen" << std::endl;

    return 0;
}
