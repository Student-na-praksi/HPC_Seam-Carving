#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <numa.h>

#include "carving.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

typedef enum {
    MODE_DYNAMIC = 0,
    MODE_GREEDY = 1,
    MODE_TRIANGLE = 2
} seam_mode_t;

void copy_image(unsigned char *image_out, const unsigned char *image_in, const size_t size)
{
    
    #pragma omp parallel
    {
        // Print thread, CPU, and NUMA node information
        #pragma omp single
        printf("Using %d threads.\n", omp_get_num_threads());

        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();
        int node = numa_node_of_cpu(cpu);

        #pragma omp critical
        printf("Thread %d -> CPU %d NUMA %d\n", tid, cpu, node);

        // Copy the image data in parallel
        #pragma omp for
        for (size_t i = 0; i < size; ++i)
        {
            image_out[i] = image_in[i];
        }
    }
    
}
// Compute Sobel energy image from input image.
// Output is grayscale-like (same value written to R,G,B if present).
// For RGB images, Sobel is computed per channel and averaged.
// For grayscale images, only one channel is used.
void calculate_energy(unsigned char *image_out, const unsigned char *image_in, int w, int h, int cpp, int rows_per_chunk)
{
    const int channels_to_use = (cpp >= 3) ? 3 : 1;

    #define CLP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
    #define PIX(ii, jj, ch) image_in[((ii) * w + (jj)) * cpp + (ch)]

    // #pragma omp parallel
    // {
    //     // Print thread, CPU, and NUMA node information
    //     #pragma omp single
    //     printf("Using %d threads.\n", omp_get_num_threads());

    //     int tid = omp_get_thread_num();
    //     int cpu = sched_getcpu();
    //     int node = numa_node_of_cpu(cpu);

    //     #pragma omp critical
    //     printf("Thread %d -> CPU %d NUMA %d\n", tid, cpu, node);
    // }

    //OUTPUT FROMATING
    // Write one scalar energy value into output pixel:
    // - grayscale: one channel
    // - RGB: write same value to R,G,B
    // - RGBA: preserve alpha
    #define STORE_ENERGY(ii, jj, evalue)                    \
        do {                                                \
            int o = ((ii) * w + (jj)) * cpp;                \
            unsigned char ev = (unsigned char)(evalue);     \
            image_out[o + 0] = ev;                          \
            if (cpp > 1) image_out[o + 1] = ev;             \
            if (cpp > 2) image_out[o + 2] = ev;             \
            if (cpp > 3) image_out[o + 3] = image_in[o + 3];\
        } while (0)

    // 1) Interior pixels: no clamp checks needed because neighbors exist on all sides.
    //    i = row index, j = column index.
    //    schedule(static, rows_per_chunk) assigns contiguous row blocks to threads.
    #pragma omp parallel for schedule(static, rows_per_chunk)
    for (int i = 1; i < h - 1; ++i) {
        const unsigned char *row_above = image_in + (i - 1) * w * cpp;
        const unsigned char *row_current = image_in + i * w * cpp;
        const unsigned char *row_below = image_in + (i + 1) * w * cpp;

        for (int j = 1; j < w - 1; ++j) {
            int e_sum = 0;

            for (int ch = 0; ch < channels_to_use; ++ch) {
                int jm = (j - 1) * cpp + ch;
                int j0 = j * cpp + ch;
                int jp = (j + 1) * cpp + ch;

                int ul = row_above[jm], u = row_above[j0], ur = row_above[jp];
                int l = row_current[jm],                  r = row_current[jp];
                int dl = row_below[jm], d = row_below[j0], dr = row_below[jp];

                int gx = -ul - 2 * l - dl + ur + 2 * r + dr;
                int gy = -ul - 2 * u - ur + dl + 2 * d + dr;

                e_sum += (int)sqrt((double)(gx * gx + gy * gy));
            }

            //per chanel adverage
            int e = e_sum / channels_to_use;
            if (e > 255) e = 255;
            STORE_ENERGY(i, j, e);
        }
    }

    // 2) Left and right border columns for each row.
    //    Clamp maps out-of-range neighbors to nearest valid pixel.
    for (int i = 0; i < h; ++i) {
        for (int edge = 0; edge < 2; ++edge) {
            int j = (edge == 0) ? 0 : (w - 1);
            int im = CLP(i - 1, 0, h - 1), ip = CLP(i + 1, 0, h - 1);
            int jm = CLP(j - 1, 0, w - 1), jp = CLP(j + 1, 0, w - 1);

            int e_sum = 0;
            for (int ch = 0; ch < channels_to_use; ++ch) {
                int ul = PIX(im, jm, ch), u = PIX(im, j, ch), ur = PIX(im, jp, ch);
                int l = PIX(i, jm, ch),                         r = PIX(i, jp, ch);
                int dl = PIX(ip, jm, ch), d = PIX(ip, j, ch), dr = PIX(ip, jp, ch);

                int gx = -ul - 2 * l - dl + ur + 2 * r + dr;
                int gy = -ul - 2 * u - ur + dl + 2 * d + dr;
                e_sum += (int)sqrt((double)(gx * gx + gy * gy));
            }

            int e = e_sum / channels_to_use;
            if (e > 255) e = 255;
            STORE_ENERGY(i, j, e);
        }
    }

    // 3) Top and bottom border rows, excluding corners.
    //    Corners were already covered by step 2.
    for (int j = 1; j < w - 1; ++j) {
        for (int edge = 0; edge < 2; ++edge) {
            int i = (edge == 0) ? 0 : (h - 1);
            int im = CLP(i - 1, 0, h - 1), ip = CLP(i + 1, 0, h - 1);
            int jm = CLP(j - 1, 0, w - 1), jp = CLP(j + 1, 0, w - 1);

            int e_sum = 0;
            for (int ch = 0; ch < channels_to_use; ++ch) {
                int ul = PIX(im, jm, ch), u = PIX(im, j, ch), ur = PIX(im, jp, ch);
                int l = PIX(i, jm, ch),                         r = PIX(i, jp, ch);
                int dl = PIX(ip, jm, ch), d = PIX(ip, j, ch), dr = PIX(ip, jp, ch);

                int gx = -ul - 2 * l - dl + ur + 2 * r + dr;
                int gy = -ul - 2 * u - ur + dl + 2 * d + dr;
                e_sum += (int)sqrt((double)(gx * gx + gy * gy));
            }

            int e = e_sum / channels_to_use;
            if (e > 255) e = 255;
            STORE_ENERGY(i, j, e);
        }
    }

    #undef STORE_ENERGY
    #undef PIX
    #undef CLP
}

void cumulative_energy_update_cell(const unsigned char *image_in, int *cumulative, signed char *steering, int i, int j, int w, int cpp) {
    int best_col = j;
        int best = cumulative[(i + 1) * w + j];

        if (j > 0) {
            int left = cumulative[(i + 1) * w + (j - 1)];
            if (left < best) {
                best = left;
                best_col = j - 1;
            }
        }
        if (j + 1 < w) {
            int right = cumulative[(i + 1) * w + (j + 1)];
            if (right < best) {
                best = right;
                best_col = j + 1;
            }
        }

        cumulative[i * w + j] = (int)image_in[(i * w + j) * cpp + 0] + best;
        steering[i * w + j] = (signed char)(best_col - j);
}

void seam_carving_dynamic(const unsigned char *image_in, int w, int h, int cpp, int rows_per_chunk, int *remove_indexes)
{
    if (w <= 1 || h <= 0 || cpp <= 0) {
        return;
    }

    // cumulative[i,j] stores minimum path energy from (i,j) down to last row.
    int *cumulative = (int *)malloc((size_t)w * (size_t)h * sizeof(int));
    // steering[i,j] stores next-row column movement: -1 (left), 0 (down), +1 (right).
    signed char *steering = (signed char *)malloc((size_t)w * (size_t)h * sizeof(signed char));

    if (cumulative == NULL || steering == NULL || remove_indexes == NULL) {
        printf("Error: Failed to allocate memory for seam carving buffers!\n");
        free(cumulative);
        free(steering);
        return;
    }

    // Base case: last row has no children, so cumulative equals local energy.
    // image_in here is the energy image; channel 0 contains the scalar energy value.
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < w; ++j) {
        cumulative[(h - 1) * w + j] = image_in[((h - 1) * w + j) * cpp + 0];
        steering[(h - 1) * w + j] = 0;
    }

    // DP recurrence (bottom -> top):
    // M(i,j) = E(i,j) + min(M(i+1,j-1), M(i+1,j), M(i+1,j+1)).
    // Rows are sequential because each row depends on the row below.
    // Columns inside one row are independent and parallelized.
    #pragma omp parallel
    {
        for (int i = h - 2; i >= 0; --i) {
            #pragma omp for schedule(static)
            for (int j = 0; j < w; ++j) {
                int best_col = j;
                int best = cumulative[(i + 1) * w + j];

                if (j > 0) {
                    int left = cumulative[(i + 1) * w + (j - 1)];
                    if (left < best) {
                        best = left;
                        best_col = j - 1;
                    }
                }
                if (j + 1 < w) {
                    int right = cumulative[(i + 1) * w + (j + 1)];
                    if (right < best) {
                        best = right;
                        best_col = j + 1;
                    }
                }

                cumulative[i * w + j] = (int)image_in[(i * w + j) * cpp + 0] + best;
                steering[i * w + j] = (signed char)(best_col - j); // -1, 0, +1
            }
        }
    }

    // Seam starts at global minimum in top row.
    int best_top_col = 0;
    int best_top_energy = cumulative[0];
    for (int j = 1; j < w; ++j) {
        int val = cumulative[j];
        if (val < best_top_energy) {
            best_top_energy = val;
            best_top_col = j;
        }
    }

    // Backtrack seam from top to bottom using stored steering directions.
    // remove_indexes[i] = seam column index in row i.
    int j = best_top_col;
    for (int i = 0; i < h; ++i) {
        remove_indexes[i] = j;
        if (i + 1 < h) {
            j += steering[i * w + j];
            if (j < 0) j = 0;
            if (j >= w) j = w - 1;
        }
    }

    free(cumulative);
    free(steering);
}

int seam_carving_greedy(const unsigned char *image_in, int w, int h, int cpp, int rows_per_chunk, int *remove_indexes, int k_remove)
{
    (void)rows_per_chunk;

    if (w <= 1 || h <= 0 || cpp <= 0 || remove_indexes == NULL || k_remove <= 0) {
        return 0;
    }

    if (k_remove > w - 1) {
        k_remove = w - 1;
    }

    int *candidate_paths = (int *)malloc((size_t)w * (size_t)h * sizeof(int));
    int *candidate_scores = (int *)malloc((size_t)w * sizeof(int));
    int *order = (int *)malloc((size_t)w * sizeof(int));

    // Builds one greedy seam candidate per top-row column in parallel.
    #pragma omp parallel for schedule(static)
    for (int start_col = 0; start_col < w; ++start_col) {
        int j = start_col;
        int score = (int)image_in[j * cpp + 0];
        candidate_paths[start_col * h + 0] = j;

        for (int i = 1; i < h; ++i) {
            int best_j = j;
            int best_e = (int)image_in[(i * w + j) * cpp + 0];

            if (j > 0) {
                int e_left = (int)image_in[(i * w + (j - 1)) * cpp + 0];
                if (e_left < best_e) {
                    best_e = e_left;
                    best_j = j - 1;
                }
            }
            if (j + 1 < w) {
                int e_right = (int)image_in[(i * w + (j + 1)) * cpp + 0];
                if (e_right < best_e) {
                    best_e = e_right;
                    best_j = j + 1;
                }
            }

            j = best_j;
            candidate_paths[start_col * h + i] = j;
            score += best_e;
        }

        candidate_scores[start_col] = score;
    }

    for (int c = 0; c < w; ++c) {
        order[c] = c;
    }

    // Insertion sort for candidates
    for (int i = 1; i < w; ++i) {
        int idx = order[i];
        int key_score = candidate_scores[idx];
        int j = i - 1;
        while (j >= 0 && candidate_scores[order[j]] > key_score) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = idx;
    }

    int accepted = 0;

    for (int oi = 0; oi < w && accepted < k_remove; ++oi) {
        int cand_col = order[oi];
        const int *cand_path = candidate_paths + cand_col * h;
        int valid = 1;

        for (int s = 0; s < accepted && valid; ++s) {
            const int *acc_path = remove_indexes + s * h;
            int relation = 0;

            for (int i = 0; i < h; ++i) {
                int d = cand_path[i] - acc_path[i];
                if (d == 0) {
                    valid = 0;
                    break;
                }

                if (relation == 0) {
                    relation = (d > 0) ? 1 : -1;
                } else if ((relation > 0 && d < 0) || (relation < 0 && d > 0)) {
                    // Crossing detected of seams detected
                    valid = 0;
                    break;
                }
            }
        }

        if (valid) {
            for (int i = 0; i < h; ++i) {
                remove_indexes[accepted * h + i] = cand_path[i];
            }
            accepted++;
        }
    }

    free(candidate_paths);
    free(candidate_scores);
    free(order);
    return accepted;
}

void seam_carving_triangle(const unsigned char *image_in, int w, int h, int cpp, int rows_per_chunk, int *remove_indexes, int strip_height)
{
    int *cumulative = (int *)malloc((size_t)w * (size_t)h * sizeof(int));
    signed char *steering = (signed char *)malloc((size_t)w * (size_t)h * sizeof(signed char));

    #pragma omp parallel for schedule(static)
    for(int j = 0; j < w; ++j) {
        cumulative[(h - 1) * w + j] = image_in[((h - 1) * w + j) * cpp + 0];
        steering[(h - 1) * w + j] = 0;
    }

    #pragma omp parallel
    for(int strip_bottom = h - 2; strip_bottom >= 0; strip_bottom -= strip_height) {
        int strip_top = strip_bottom - strip_height + 1;
        if (strip_top < 0) strip_top = 0;
        int current_strip_height = strip_bottom - strip_top + 1;

        // We calculate the up-facing triangles
        #pragma omp for schedule(static)
        for(int col = current_strip_height - 1; col < w + current_strip_height; col += 2 * current_strip_height ) {
            for(int i = strip_bottom; i >= strip_top; --i) {
                int triangle_span = i - strip_top;
                int row_start = col - triangle_span;
                int row_end = col + triangle_span;
                
                if (row_start < 0) row_start = 0;
                if (row_end >= w) row_end = w - 1;
                if (row_start > row_end) continue;

                for (int j = row_start; j <= row_end; ++j) {
                    cumulative_energy_update_cell(image_in, cumulative, steering, i, j, w, cpp);
                }
            }
        }

        // We calculate the down-facing triangles
        #pragma omp for schedule(static)
        for(int col = -1; col < w + current_strip_height; col += 2 * current_strip_height) {
            for(int i = strip_bottom; i >= strip_top; --i) {
                int triangle_span = strip_bottom - i;
                int row_start = col - triangle_span;
                int row_end = col + triangle_span;

                if (row_start < 0) row_start = 0;
                if (row_end >= w) row_end = w - 1;
                if (row_start > row_end) continue;

                for(int j = row_start; j <= row_end; ++j) {
                    cumulative_energy_update_cell(image_in, cumulative, steering, i, j, w, cpp);
                }
            }
        }
    }

    int best_top_col = 0;
    int best_top_energy = cumulative[0];

    for (int j = 1; j < w; ++j) {
        int val = cumulative[j];
        if (val < best_top_energy) {
            best_top_energy = val;
            best_top_col = j;
        }
    }

    int j = best_top_col;
    for (int i = 0; i < h; ++i) {
        remove_indexes[i] = j;
        if (i + 1 < h) {
            j += steering[i * w + j];
            if (j < 0) j = 0;
            if (j >= w) j = w - 1;
        }
    }

    free(cumulative);
    free(steering);
}

void remove_seams(unsigned char *image_out, const unsigned char *image_in, int in_w, int h, int cpp, int rows_per_chunk, const int *remove_indexes)
{
    if (in_w <= 1 || h <= 0 || cpp <= 0 || remove_indexes == NULL) {
        return;
    }
    const int out_w = in_w - 1;

    // Remove one vertical seam per row.
    // Read from input rows with stride in_w and write compact output rows with stride out_w.
    // This keeps the next iteration's memory layout consistent with active width.
    #pragma omp parallel for schedule(static, rows_per_chunk)
    for (int i = 0; i < h; ++i) {
        int s = remove_indexes[i];

        if (s < 0) s = 0;
        if (s >= in_w) s = in_w - 1;

        for (int col = 0; col < out_w; ++col) {
            int src_col = (col < s) ? col : (col + 1);
            int src = (i * in_w + src_col) * cpp;
            int dst = (i * out_w + col) * cpp;
            for (int ch = 0; ch < cpp; ++ch) {
                image_out[dst + ch] = image_in[src + ch];
            }
        }
    }
}

void remove_seams_multi(unsigned char *image_out, const unsigned char *image_in, int in_w, int h, int cpp, int rows_per_chunk, const int *remove_indexes, int seam_count)
{
    if (in_w <= 1 || h <= 0 || cpp <= 0 || remove_indexes == NULL || seam_count <= 0) {
        return;
    }

    if (seam_count > in_w - 1) {
        seam_count = in_w - 1;
    }

    const int out_w = in_w - seam_count;

    #pragma omp parallel for schedule(static, rows_per_chunk)
    for (int i = 0; i < h; ++i) {
        int *row_seams = (int *)malloc((size_t)seam_count * sizeof(int));

        for (int s = 0; s < seam_count; ++s) {
            int col = remove_indexes[s * h + i];
            row_seams[s] = col;
        }

        // Insertion sort so that the removal is from left to right
        for (int a = 1; a < seam_count; ++a) {
            int key = row_seams[a];
            int b = a - 1;
            while (b >= 0 && row_seams[b] > key) {
                row_seams[b + 1] = row_seams[b];
                --b;
            }
            row_seams[b + 1] = key;
        }

        int skip_idx = 0;
        int out_col = 0;

        for (int in_col = 0; in_col < in_w; ++in_col) {
            if (skip_idx < seam_count && in_col == row_seams[skip_idx]) {
                skip_idx++;
                continue;
            }

            int src = (i * in_w + in_col) * cpp;
            int dst = (i * out_w + out_col) * cpp;
            for (int ch = 0; ch < cpp; ++ch) {
                image_out[dst + ch] = image_in[src + ch];
            }
            out_col++;
        }

        free(row_seams);
    }
}

static int estimate_rows_per_chunk(int active_w, int cpp)
{
    size_t row_bytes = (size_t)active_w * (size_t)cpp;
    size_t target_cache = (size_t)(0.6 * 512 * 1024);
    int rows_per_chunk = (int)(target_cache / (4 * row_bytes));
    if (rows_per_chunk < 1) rows_per_chunk = 1;
    return rows_per_chunk;
}

static void run_dynamic_mode(
    unsigned char **current_in,
    unsigned char **current_out,
    unsigned char *image_energy,
    int h,
    int cpp,
    int seam_number,
    int *active_w)
{
    int *remove_indexes = (int *)malloc((size_t)h * sizeof(int));

    for (int iter = 0; iter < seam_number; ++iter) {
        int rows_per_chunk = estimate_rows_per_chunk(*active_w, cpp);

        calculate_energy(image_energy, *current_in, *active_w, h, cpp, rows_per_chunk);

        seam_carving_dynamic(image_energy, *active_w, h, cpp, rows_per_chunk, remove_indexes);

        remove_seams(*current_out, *current_in, *active_w, h, cpp, rows_per_chunk, remove_indexes);

        unsigned char *tmp = *current_in;
        *current_in = *current_out;
        *current_out = tmp;
        (*active_w)--;
    }

    free(remove_indexes);
}

static void run_greedy_mode(unsigned char **current_in, unsigned char **current_out, unsigned char *image_energy, int h, int cpp, int seam_number, int batch_size, int *active_w)
{
    if (batch_size < 1) {
        batch_size = 1;
    }

    int *remove_indexes = (int *)malloc((size_t)h * (size_t)batch_size * sizeof(int));

    int removed_total = 0;
    int iter = 0;

    while (removed_total < seam_number) {
        iter++;
        int rows_per_chunk = estimate_rows_per_chunk(*active_w, cpp);

        int seams_this_iter = batch_size;
        int remaining = seam_number - removed_total;
        if (seams_this_iter > remaining) seams_this_iter = remaining;
        if (seams_this_iter > *active_w - 1) seams_this_iter = *active_w - 1;
        if (seams_this_iter < 1) seams_this_iter = 1;

        calculate_energy(image_energy, *current_in, *active_w, h, cpp, rows_per_chunk);

        int selected_seams = seam_carving_greedy(image_energy, *active_w, h, cpp, rows_per_chunk, remove_indexes, seams_this_iter);
        if (selected_seams < 1) selected_seams = 1;
        if (selected_seams > *active_w - 1) selected_seams = *active_w - 1;

        remove_seams_multi(*current_out, *current_in, *active_w, h, cpp, rows_per_chunk, remove_indexes, selected_seams);

        unsigned char *tmp = *current_in;
        *current_in = *current_out;
        *current_out = tmp;
        *active_w -= selected_seams;
        removed_total += selected_seams;
    }

    free(remove_indexes);
}

static void run_triangle_mode(unsigned char **current_in, unsigned char **current_out, unsigned char *image_energy, int h, int cpp, int seam_number, int *active_w, int strip_height)
{
    int *remove_indexes = (int *)malloc((size_t)h * sizeof(int));

    for (int iter = 0; iter < seam_number; ++iter) {
        int rows_per_chunk = estimate_rows_per_chunk(*active_w, cpp);

        calculate_energy(image_energy, *current_in, *active_w, h, cpp, rows_per_chunk);

        seam_carving_triangle(image_energy, *active_w, h, cpp, rows_per_chunk, remove_indexes, strip_height);

        remove_seams(*current_out, *current_in, *active_w, h, cpp, rows_per_chunk, remove_indexes);

        unsigned char *tmp = *current_in;
        *current_in = *current_out;
        *current_out = tmp;
        (*active_w)--;
    }

    free(remove_indexes);
}

#ifndef CARVING_NO_MAIN
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];
    int seam_number = 1;
    seam_mode_t mode = MODE_DYNAMIC;
    int batch_size = 1;
    int strip_height = 16;
    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);

    for (int argi = 3; argi < argc; ++argi) {
        if (!strcmp(argv[argi], "--seam_number")) {
            if (argi + 1 >= argc) {
                printf("Error: Missing value for --seam_number\n");
                exit(EXIT_FAILURE);
            }
            seam_number = atoi(argv[++argi]);
            if (seam_number < 1) {
                printf("Error: --seam_number must be >= 1\n");
                exit(EXIT_FAILURE);
            }
        } else if (!strcmp(argv[argi], "--mode")) {
            if (argi + 1 >= argc) {
                printf("Error: Missing value for --mode\n");
                exit(EXIT_FAILURE);
            }
            const char *mode_arg = argv[++argi];
            if (!strcmp(mode_arg, "dynamic")) {
                mode = MODE_DYNAMIC;
            } else if (!strcmp(mode_arg, "greedy")) {
                mode = MODE_GREEDY;
            } else if (!strcmp(mode_arg, "triangle")) {
                mode = MODE_TRIANGLE;
            } else {
                printf("Error: Unknown mode %s (expected dynamic|greedy|triangle)\n", mode_arg);
                exit(EXIT_FAILURE);
            }
        } else if (!strcmp(argv[argi], "--batch_size")) {
            if (argi + 1 >= argc) {
                printf("Error: Missing value for --batch_size\n");
                exit(EXIT_FAILURE);
            }
            batch_size = atoi(argv[++argi]);
            if (batch_size < 1) {
                printf("Error: --batch_size must be >= 1\n");
                exit(EXIT_FAILURE);
            }
        } else if (!strcmp(argv[argi], "--strip_height")) {
            if (argi + 1 >= argc) {
                printf("Error: Missing value for --strip_height\n");
                exit(EXIT_FAILURE);
            }
            strip_height = atoi(argv[++argi]);
            if (strip_height < 1) {
                printf("Error: --strip_height must be >= 1\n");
                exit(EXIT_FAILURE);
            }
        } else {
            printf("Error: Unknown option %s\n", argv[argi]);
            printf("USAGE: carving input_image output_image [--seam_number N] [--mode dynamic|greedy|triangle] [--batch_size K] [--strip_height SH]\n");
            exit(EXIT_FAILURE);
        }
    }

    // Load image from file and allocate space for the output image
    int w, h, cpp;
    unsigned char *image_loaded = stbi_load(image_in_name, &w, &h, &cpp, COLOR_CHANNELS);

    const size_t datasize = w * h * cpp * sizeof(unsigned char);
    unsigned char *image_a = (unsigned char *)malloc(datasize);
    unsigned char *image_b = (unsigned char *)malloc(datasize);
    unsigned char *image_energy = (unsigned char *)malloc(datasize);

    // Copy the input image into output and mesure execution time
    // copy_image(image_out, image_in, datasize);
    memcpy(image_a, image_loaded, datasize);
    stbi_image_free(image_loaded);

    if (seam_number > w - 1) {
        seam_number = w - 1;
    }

    unsigned char *current_in = image_a;
    unsigned char *current_out = image_b;
    int active_w = w;

    if (mode == MODE_GREEDY) {
        run_greedy_mode(&current_in, &current_out, image_energy, h, cpp, seam_number, batch_size, &active_w);
    } else if(mode == MODE_TRIANGLE) {
        run_triangle_mode(&current_in, &current_out, image_energy, h, cpp, seam_number, &active_w, strip_height);
    } else {
        run_dynamic_mode(&current_in, &current_out, image_energy, h, cpp, seam_number, &active_w);
    }

    // Write the output image to file
    const char *file_type = strrchr(image_out_name, '.');  
    file_type++; // skip the dot

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, active_w, h, cpp, current_in, active_w * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, active_w, h, cpp, current_in, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, active_w, h, cpp, current_in);
    else
        printf("Error: Unknown image format %s\n", file_type);

    int out_w = 0, out_h = 0, out_cpp = 0;
    unsigned char *verify_image = stbi_load(image_out_name, &out_w, &out_h, &out_cpp, 0);
    if (verify_image != NULL) {
        stbi_image_free(verify_image);
    }

    free(image_a);
    free(image_b);
    free(image_energy);

    return 0;
}
#endif