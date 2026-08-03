// Monte Carlo portfolio simulator (Phase 4).
//
// Correlated annual returns r = mu + L z (z std normal, L = chol(Sigma)),
// portfolio value compounded over HORIZON years. Interface: CSV in
// (outputs/phase4: expected_returns.csv, covariance.csv, weights.csv),
// summary CSV out — a deliberate subprocess/CSV design (simpler and more
// auditable than pybind11 for a batch engine; benchmarked against the
// NumPy reference for correctness first, speed second).
//
// Build: clang++ -O3 -std=c++17 -o mc_engine mc_engine.cpp
// Run:   ./mc_engine <n_paths> <seed> <outputs_dir>

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static const int HORIZON = 5;

using Mat = std::vector<std::vector<double>>;

static std::vector<std::vector<std::string>> read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(f, line)) {
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) cells.push_back(cell);
        rows.push_back(cells);
    }
    return rows;
}

static Mat cholesky(const Mat& a) {
    size_t n = a.size();
    Mat l(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j <= i; ++j) {
            double s = a[i][j];
            for (size_t k = 0; k < j; ++k) s -= l[i][k] * l[j][k];
            l[i][j] = (i == j) ? std::sqrt(s) : s / l[j][j];
        }
    return l;
}

int main(int argc, char** argv) {
    long n_paths = argc > 1 ? std::atol(argv[1]) : 200000;
    unsigned seed = argc > 2 ? (unsigned)std::atoi(argv[2]) : 42;
    std::string dir = argc > 3 ? argv[3] : "outputs/phase4";

    auto er_rows = read_csv(dir + "/expected_returns.csv");
    std::vector<double> mu;
    // header: club,...,exp_return (last column)
    for (size_t i = 1; i < er_rows.size(); ++i)
        mu.push_back(std::stod(er_rows[i].back()));
    size_t n = mu.size();

    auto cov_rows = read_csv(dir + "/covariance.csv");
    Mat cov(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            cov[i][j] = std::stod(cov_rows[i + 1][j]);  // row 0 = header

    auto w_rows = read_csv(dir + "/weights.csv");
    std::vector<double> w;
    for (size_t i = 1; i < w_rows.size(); ++i)
        w.push_back(std::stod(w_rows[i][1]));  // col 1 = max_sharpe

    Mat L = cholesky(cov);
    // Algebraic reduction (used identically in the NumPy reference): the
    // portfolio return w.(mu + Lz) = w.mu + (L^T w).z — one dot product per
    // year instead of an O(n^2) matrix-vector product. Distributionally
    // exact for a fixed-weight portfolio.
    double mu_p = 0.0;
    for (size_t i = 0; i < n; ++i) mu_p += w[i] * mu[i];
    std::vector<double> lw(n, 0.0);
    for (size_t k = 0; k < n; ++k)
        for (size_t i = k; i < n; ++i) lw[k] += L[i][k] * w[i];

    // Single-threaded honesty note (kept for the write-up): NumPy's
    // ziggurat+SIMD normal generation is genuinely fast — std::mt19937_64/
    // std::normal_distribution lost 0.97s vs 0.52s at 1M paths, and
    // xoshiro256++/Box-Muller only reached parity (0.76s). The C++ engine's
    // real edge is embarrassingly-parallel path splitting across cores,
    // deterministic given (seed, thread count).
    unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<double> values(n_paths);

    auto worker = [&](unsigned tid, long lo, long hi) {
        uint64_t sd = seed + 0x9E3779B97F4A7C15ULL * (tid + 1);
        uint64_t s[4] = {sd ^ 0x9E3779B97F4A7C15ULL, sd + 0xBF58476D1CE4E5B9ULL,
                         sd * 0x94D049BB133111EBULL + 1, sd + 0x2545F4914F6CDD1DULL};
        auto rotl = [](uint64_t x, int k) { return (x << k) | (x >> (64 - k)); };
        auto next = [&]() {
            uint64_t r = rotl(s[0] + s[3], 23) + s[0];
            uint64_t t = s[1] << 17;
            s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
            s[3] = rotl(s[3], 45);
            return r;
        };
        auto uniform = [&]() {
            return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        };
        bool have_spare = false; double spare = 0.0;
        auto normal = [&]() {
            if (have_spare) { have_spare = false; return spare; }
            double u = uniform(), v2 = uniform();
            double m = std::sqrt(-2.0 * std::log(u));
            spare = m * std::sin(6.283185307179586 * v2);
            have_spare = true;
            return m * std::cos(6.283185307179586 * v2);
        };
        for (long p = lo; p < hi; ++p) {
            double v = 1.0;
            for (int t = 0; t < HORIZON; ++t) {
                double shock = 0.0;
                for (size_t k = 0; k < n; ++k) shock += lw[k] * normal();
                v *= 1.0 + mu_p + shock;
            }
            values[p] = v;
        }
    };

    std::vector<std::thread> pool;
    long chunk = n_paths / n_threads;
    for (unsigned t = 0; t < n_threads; ++t) {
        long lo = t * chunk;
        long hi = (t == n_threads - 1) ? n_paths : lo + chunk;
        pool.emplace_back(worker, t, lo, hi);
    }
    for (auto& th : pool) th.join();

    std::sort(values.begin(), values.end());
    auto q = [&](double a) { return values[(size_t)(a * (n_paths - 1))]; };
    double mean = 0; for (double v : values) mean += v; mean /= n_paths;
    long losses = 0; for (double v : values) if (v < 1.0) ++losses;

    std::ofstream out(dir + "/mc_cpp_summary.csv");
    out << "mean,p5,p50,p95,prob_loss,paths\n"
        << mean << "," << q(0.05) << "," << q(0.50) << "," << q(0.95) << ","
        << (double)losses / n_paths << "," << n_paths << "\n";
    std::cout << "mean=" << mean << " p5=" << q(0.05) << " p50=" << q(0.50)
              << " p95=" << q(0.95) << " prob_loss=" << (double)losses / n_paths
              << "\n";
    return 0;
}
