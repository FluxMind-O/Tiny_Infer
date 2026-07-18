#include "tinyinfer/common.h"
#include <algorithm>
#include <chrono>

namespace tinynfer {

double get_wall_time_us() {
    static auto start = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(now - start).count();
}

void print_tensor(const dtype* h_data, int n, const std::string& name,
                  int max_print) {
    printf("%s = [", name.c_str());
    int p = std::min(n, max_print);
    for (int i = 0; i < p; ++i) printf("%.4f%s", h_data[i], i + 1 < p ? ", " : "");
    if (n > p) printf(", ...(%d total)", n);
    printf("]\n");
}

}  // namespace tinynfer
