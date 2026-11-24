#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>
#include "ufc.h"

namespace {
constexpr int cell_dofs = 15;
constexpr int macro_dofs = 30;

struct Task {
    int kind;
    int first;
    int second;
    int face0;
    int face1;
    int marker;
};

struct Assembly {
    std::unique_ptr<ufc::form> form;
    std::array<std::unique_ptr<ufc::cell_integral>, 2> cell;
    std::array<std::unique_ptr<ufc::exterior_facet_integral>, 5> exterior;
    std::unique_ptr<ufc::interior_facet_integral> interior;
    std::vector<std::array<double, macro_dofs>> input;
    std::vector<const double*> coefficients;
    int rank;

    explicit Assembly(ufc::form* handle)
        : form(handle), input(form->num_coefficients()), coefficients(input.size()),
          rank(form->rank()) {
        for (std::size_t i = 0; i < cell.size(); ++i) {
            cell[i].reset(form->create_cell_integral(i));
        }
        for (std::size_t i = 0; i < exterior.size(); ++i) {
            exterior[i].reset(form->create_exterior_facet_integral(i));
        }
        interior.reset(form->create_interior_facet_integral(1));
        for (std::size_t i = 0; i < input.size(); ++i) {
            coefficients[i] = input[i].data();
        }
    }

    void fill(const int* kinds, const double* scalar, const double* now,
              const double* previous, const int* dof0, const int* dof1,
              const std::vector<bool>& enabled) {
        for (std::size_t k = 0; k < input.size(); ++k) {
            if (!enabled[k]) continue;
            auto& data = input[k];
            if (kinds[k] == 0) {
                data[0] = data[1] = scalar[k];
            } else {
                const double* state = kinds[k] == 1 ? now : previous;
                for (int i = 0; i < cell_dofs; ++i) {
                    data[i] = state[dof0[i]];
                    if (dof1) data[cell_dofs + i] = state[dof1[i]];
                }
            }
        }
    }
};
}

extern "C" {
void* create_assembly(void* form) {
    return new Assembly(static_cast<ufc::form*>(form));
}

void destroy_assembly(void* assembly) {
    delete static_cast<Assembly*>(assembly);
}

int coefficient_count(void* assembly) {
    return static_cast<Assembly*>(assembly)->input.size();
}

int coefficient_position(void* assembly, int index) {
    return static_cast<Assembly*>(assembly)->form->original_coefficient_position(index);
}

void assemble_batch(void* handle, int count, const int* tasks, const double* coordinates,
                    const int* dofmap, const int* coefficient_kinds, const double* constants,
                    const double* now, const double* previous, const int* destination,
                    double* result) {
    auto& a = *static_cast<Assembly*>(handle);
    std::array<double, macro_dofs * macro_dofs> tensor;
    for (int j = 0; j < count; ++j) {
        const int* row = tasks + 6 * j;
        const Task t{row[0], row[1], row[2], row[3], row[4], row[5]};
        const double* x = coordinates + 6 * t.first;
        const int* d = dofmap + cell_dofs * t.first;
        const auto** w = a.coefficients.data();
        int size = cell_dofs;
        bool available = true;
        if (t.kind == 0) {
            auto* kernel = a.cell[t.marker].get();
            available = kernel != nullptr;
            if (kernel) {
                a.fill(coefficient_kinds, constants, now, previous, d, nullptr,
                       kernel->enabled_coefficients());
                kernel->tabulate_tensor(tensor.data(), w, x, 0);
            }
        } else if (t.kind == 1) {
            auto* kernel = a.exterior[t.marker].get();
            available = kernel != nullptr;
            if (kernel) {
                a.fill(coefficient_kinds, constants, now, previous, d, nullptr,
                       kernel->enabled_coefficients());
                kernel->tabulate_tensor(tensor.data(), w, x, t.face0, 0);
            }
        } else {
            size = macro_dofs;
            auto* kernel = a.interior.get();
            available = kernel != nullptr;
            if (kernel) {
                a.fill(coefficient_kinds, constants, now, previous, d,
                       dofmap + cell_dofs * t.second, kernel->enabled_coefficients());
                kernel->tabulate_tensor(tensor.data(), w, x, coordinates + 6 * t.second,
                                        t.face0, t.face1, 0, 0);
            }
        }
        int width = a.rank == 1 ? size : size * size;
        if (available) {
            for (int i = 0; i < width; ++i) {
                const int offset = destination[i];
                if (offset >= 0) result[offset] += tensor[i];
            }
        }
        destination += width;
    }
}
}
