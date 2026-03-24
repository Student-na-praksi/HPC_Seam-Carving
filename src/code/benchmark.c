#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "carving.h"
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 0

#define EXPERIMENT_IMAGES_DIR "../test_images"
#define EXPERIMENT_OUTPUT_CSV "dynamic_thread_experiment.csv"
#define EXPERIMENT_SEAM_NUMBER 32
#define EXPERIMENT_THREAD_MIN 1
#define EXPERIMENT_THREAD_STEP 1
#define EXPERIMENT_IMAGE_COUNT 5
#define EXPERIMENT_ITERATION 5

static const char *EXPERIMENT_IMAGE_FILES[] = {
    "720x480.png",
    "1024x768.png",
    "1920x1200.png",
    "3840x2160.png",
    "7680x4320.png"
};

typedef struct {
    double energy;
    double seam;
    double remove;
    double total;
} timing_result_t;

static int estimate_rows_per_chunk(int active_w, int cpp)
{
    size_t row_bytes = (size_t)active_w * (size_t)cpp;
    size_t target_cache = (size_t)(0.6 * 512 * 1024);
    int rows_per_chunk = (int)(target_cache / (4 * row_bytes));
    if (rows_per_chunk < 1) rows_per_chunk = 1;
    return rows_per_chunk;
}

static int run_dynamic_benchmark_once(const unsigned char *image_src, int w, int h, int cpp, int seam_number, int energy_threads, int seam_threads, int remove_threads, timing_result_t *result)
{
    const size_t datasize = (size_t)w * (size_t)h * (size_t)cpp * sizeof(unsigned char);
    unsigned char *image_a = (unsigned char *)malloc(datasize);
    unsigned char *image_b = (unsigned char *)malloc(datasize);
    unsigned char *image_energy = (unsigned char *)malloc(datasize);
    int *remove_indexes = (int *)malloc((size_t)h * sizeof(int));

    memcpy(image_a, image_src, datasize);

    unsigned char *current_in = image_a;
    unsigned char *current_out = image_b;
    int active_w = w;

    const int prev_dynamic = omp_get_dynamic();
    const int prev_threads = omp_get_max_threads();
    omp_set_dynamic(0);

    result->energy = 0.0;
    result->seam = 0.0;
    result->remove = 0.0;

    for (int iter = 0; iter < seam_number; ++iter) {
        int rows_per_chunk = estimate_rows_per_chunk(active_w, cpp);

        omp_set_num_threads(energy_threads);
        double start_energy = omp_get_wtime();
        // Optimised version
        calculate_energy(image_energy, current_in, active_w, h, cpp, rows_per_chunk);
        // Non optimised version
        // calculate_energy_basic(image_energy, current_in, active_w, h, cpp, rows_per_chunk);
        double stop_energy = omp_get_wtime();

        omp_set_num_threads(seam_threads);
        double start_seam = omp_get_wtime();
        seam_carving_dynamic(image_energy, active_w, h, cpp, rows_per_chunk, remove_indexes);
        double stop_seam = omp_get_wtime();

        omp_set_num_threads(remove_threads);
        double start_remove = omp_get_wtime();
        remove_seams(current_out, current_in, active_w, h, cpp, rows_per_chunk, remove_indexes);
        double stop_remove = omp_get_wtime();

        result->energy += stop_energy - start_energy;
        result->seam += stop_seam - start_seam;
        result->remove += stop_remove - start_remove;

        unsigned char *tmp = current_in;
        current_in = current_out;
        current_out = tmp;
        active_w--;
    }

    result->total = result->energy + result->seam + result->remove;

    omp_set_num_threads(prev_threads);
    omp_set_dynamic(prev_dynamic);

    free(image_a);
    free(image_b);
    free(image_energy);
    free(remove_indexes);

    return 1;
}

static int run_dynamic_thread_experiment(const char *images_dir, int seam_number, int thread_min, int thread_max, int thread_step, const char *csv_path)
{
    FILE *csv = fopen(csv_path, "w");

    fprintf(csv, "image,width,height,channels,seams,energy_threads,seam_threads,remove_threads,energy_s,seam_s,remove_s,total_s\n");

    for (int idx = 0; idx < EXPERIMENT_IMAGE_COUNT; ++idx) {
        char image_path[512];
        snprintf(image_path, sizeof(image_path), "%s/%s", images_dir, EXPERIMENT_IMAGE_FILES[idx]);

        int w = 0, h = 0, cpp = 0;
        unsigned char *loaded = stbi_load(image_path, &w, &h, &cpp, COLOR_CHANNELS);

        int seams_for_image = seam_number;
        if (seams_for_image > w - 1) seams_for_image = w - 1;

        double best_energy = 1e300;
        double best_seam = 1e300;
        double best_remove = 1e300;
        double best_total = 1e300;
        int best_energy_threads = thread_min;
        int best_seam_threads = thread_min;
        int best_remove_threads = thread_min;

        const int base_threads = thread_max;

        for (int threads = thread_min; threads <= thread_max; threads *= 2) {
            timing_result_t tr;
            run_dynamic_benchmark_once(loaded, w, h, cpp, seams_for_image, threads, base_threads, base_threads, &tr);
            if (tr.energy < best_energy) {
                best_energy = tr.energy;
                best_energy_threads = threads;
            }
        }

        for (int threads = thread_min; threads <= thread_max; threads *= 2) {
            timing_result_t tr;
            run_dynamic_benchmark_once(loaded, w, h, cpp, seams_for_image, base_threads, threads, base_threads, &tr);
            if (tr.seam < best_seam) {
                best_seam = tr.seam;
                best_seam_threads = threads;
            }
        }

        for (int threads = thread_min; threads <= thread_max; threads *= 2) {
            timing_result_t tr;
            run_dynamic_benchmark_once(loaded, w, h, cpp, seams_for_image, base_threads, base_threads, threads, &tr);
            if (tr.remove < best_remove) {
                best_remove = tr.remove;
                best_remove_threads = threads;
            }
        }

        {
            timing_result_t tr_best;
            if (run_dynamic_benchmark_once(loaded, w, h, cpp, seams_for_image, best_energy_threads, best_seam_threads, best_remove_threads, &tr_best)) {
                best_total = tr_best.total;
                fprintf(csv, "%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n", EXPERIMENT_IMAGE_FILES[idx], w, h, cpp, seams_for_image, best_energy_threads, best_seam_threads, best_remove_threads, tr_best.energy, tr_best.seam, tr_best.remove, tr_best.total);
            }
        }

        stbi_image_free(loaded);
    }

    fclose(csv);
    return 1;
}

typedef struct {
    double runtime;
    int final_width;
} greedy_vs_dynamic_result_t;

static int run_greedy_benchmark_once(const unsigned char *image_src, int w, int h, int cpp, int seam_number, int batch_size, int num_threads, unsigned char **output_image, greedy_vs_dynamic_result_t *result)
{
    const size_t datasize = (size_t)w * (size_t)h * (size_t)cpp * sizeof(unsigned char);
    unsigned char *image_a = (unsigned char *)malloc(datasize);
    unsigned char *image_b = (unsigned char *)malloc(datasize);
    unsigned char *image_energy = (unsigned char *)malloc(datasize);
    int *remove_indexes = (int *)malloc((size_t)h * (size_t)batch_size * sizeof(int));

    memcpy(image_a, image_src, datasize);

    unsigned char *current_in = image_a;
    unsigned char *current_out = image_b;
    int active_w = w;

    const int prev_dynamic = omp_get_dynamic();
    const int prev_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    double start_total = omp_get_wtime();

    int removed_total = 0;
    while (removed_total < seam_number) {
        int rows_per_chunk = estimate_rows_per_chunk(active_w, cpp);

        int seams_this_iter = batch_size;
        int remaining = seam_number - removed_total;
        if (seams_this_iter > remaining) seams_this_iter = remaining;
        if (seams_this_iter > active_w - 1) seams_this_iter = active_w - 1;
        if (seams_this_iter < 1) seams_this_iter = 1;

        calculate_energy(image_energy, current_in, active_w, h, cpp, rows_per_chunk);
        int accepted = seam_carving_greedy(image_energy, active_w, h, cpp, rows_per_chunk, remove_indexes, seams_this_iter);
        if (accepted > 0) {
            remove_seams_multi(current_out, current_in, active_w, h, cpp, rows_per_chunk, remove_indexes, accepted);
            unsigned char *tmp = current_in;
            current_in = current_out;
            current_out = tmp;
            active_w -= accepted;
            removed_total += accepted;
        } else {
            break;
        }
    }

    double stop_total = omp_get_wtime();
    result->runtime = stop_total - start_total;
    result->final_width = active_w;

    const size_t final_datasize = (size_t)active_w * (size_t)h * (size_t)cpp * sizeof(unsigned char);
    *output_image = (unsigned char *)malloc(final_datasize);
    memcpy(*output_image, current_in, final_datasize);

    omp_set_num_threads(prev_threads);
    omp_set_dynamic(prev_dynamic);

    free(image_a);
    free(image_b);
    free(image_energy);
    free(remove_indexes);

    return 1;
}

static int run_dynamic_comparison_once(const unsigned char *image_src, int w, int h, int cpp, int seam_number, int energy_threads, int seam_threads, int remove_threads, unsigned char **output_image, timing_result_t *result)
{
    const size_t datasize = (size_t)w * (size_t)h * (size_t)cpp * sizeof(unsigned char);
    unsigned char *image_a = (unsigned char *)malloc(datasize);
    unsigned char *image_b = (unsigned char *)malloc(datasize);
    unsigned char *image_energy = (unsigned char *)malloc(datasize);
    int *remove_indexes = (int *)malloc((size_t)h * sizeof(int));

    memcpy(image_a, image_src, datasize);

    unsigned char *current_in = image_a;
    unsigned char *current_out = image_b;
    int active_w = w;

    const int prev_dynamic = omp_get_dynamic();
    const int prev_threads = omp_get_max_threads();
    omp_set_dynamic(0);

    result->energy = 0.0;
    result->seam = 0.0;
    result->remove = 0.0;

    for (int iter = 0; iter < seam_number; ++iter) {
        int rows_per_chunk = estimate_rows_per_chunk(active_w, cpp);

        omp_set_num_threads(energy_threads);
        double start_energy = omp_get_wtime();
        calculate_energy(image_energy, current_in, active_w, h, cpp, rows_per_chunk);
        double stop_energy = omp_get_wtime();

        omp_set_num_threads(seam_threads);
        double start_seam = omp_get_wtime();
        seam_carving_dynamic(image_energy, active_w, h, cpp, rows_per_chunk, remove_indexes);
        double stop_seam = omp_get_wtime();

        omp_set_num_threads(remove_threads);
        double start_remove = omp_get_wtime();
        remove_seams(current_out, current_in, active_w, h, cpp, rows_per_chunk, remove_indexes);
        double stop_remove = omp_get_wtime();

        result->energy += stop_energy - start_energy;
        result->seam += stop_seam - start_seam;
        result->remove += stop_remove - start_remove;

        unsigned char *tmp = current_in;
        current_in = current_out;
        current_out = tmp;
        active_w--;
    }

    result->total = result->energy + result->seam + result->remove;

    const size_t final_datasize = (size_t)active_w * (size_t)h * (size_t)cpp * sizeof(unsigned char);
    *output_image = (unsigned char *)malloc(final_datasize);
    memcpy(*output_image, current_in, final_datasize);

    omp_set_num_threads(prev_threads);
    omp_set_dynamic(prev_dynamic);

    free(image_a);
    free(image_b);
    free(image_energy);
    free(remove_indexes);

    return 1;
}

static int run_triangle_comparison_once(const unsigned char *image_src, int w, int h, int cpp, int seam_number, int energy_threads, int seam_threads, int remove_threads, int strip_height, unsigned char **output_image, timing_result_t *result)
{
    const size_t datasize = (size_t)w * (size_t)h * (size_t)cpp * sizeof(unsigned char);
    unsigned char *image_a = (unsigned char *)malloc(datasize);
    unsigned char *image_b = (unsigned char *)malloc(datasize);
    unsigned char *image_energy = (unsigned char *)malloc(datasize);
    int *remove_indexes = (int *)malloc((size_t)h * sizeof(int));

    memcpy(image_a, image_src, datasize);

    unsigned char *current_in = image_a;
    unsigned char *current_out = image_b;
    int active_w = w;

    const int prev_dynamic = omp_get_dynamic();
    const int prev_threads = omp_get_max_threads();
    omp_set_dynamic(0);

    result->energy = 0.0;
    result->seam = 0.0;
    result->remove = 0.0;

    for (int iter = 0; iter < seam_number; ++iter) {
        int rows_per_chunk = estimate_rows_per_chunk(active_w, cpp);

        omp_set_num_threads(energy_threads);
        double start_energy = omp_get_wtime();
        calculate_energy(image_energy, current_in, active_w, h, cpp, rows_per_chunk);
        double stop_energy = omp_get_wtime();

        omp_set_num_threads(seam_threads);
        double start_seam = omp_get_wtime();
        seam_carving_triangle(image_energy, active_w, h, cpp, rows_per_chunk, remove_indexes, strip_height);
        double stop_seam = omp_get_wtime();

        omp_set_num_threads(remove_threads);
        double start_remove = omp_get_wtime();
        remove_seams(current_out, current_in, active_w, h, cpp, rows_per_chunk, remove_indexes);
        double stop_remove = omp_get_wtime();

        result->energy += stop_energy - start_energy;
        result->seam += stop_seam - start_seam;
        result->remove += stop_remove - start_remove;

        unsigned char *tmp = current_in;
        current_in = current_out;
        current_out = tmp;
        active_w--;
    }

    result->total = result->energy + result->seam + result->remove;

    const size_t final_datasize = (size_t)active_w * (size_t)h * (size_t)cpp * sizeof(unsigned char);
    *output_image = (unsigned char *)malloc(final_datasize);
    memcpy(*output_image, current_in, final_datasize);

    omp_set_num_threads(prev_threads);
    omp_set_dynamic(prev_dynamic);

    free(image_a);
    free(image_b);
    free(image_energy);
    free(remove_indexes);

    return 1;
}

static int run_dynamic_vs_triangle_experiment(const char *images_dir, int seam_number, int num_threads, int strip_height, const char *csv_path)
{
    FILE *csv = fopen(csv_path, "w");

    char results_dir[256];
    snprintf(results_dir, sizeof(results_dir), "dynamic_vs_triangle_results");
    system("mkdir -p dynamic_vs_triangle_results");

    fprintf(csv, "image,width,height,channels,seams,strip_height,dynamic_energy_s,dynamic_seam_s,dynamic_remove_s,dynamic_total_s,triangle_energy_s,triangle_seam_s,triangle_remove_s,triangle_total_s,total_speedup_dynamic_over_triangle,seam_speedup_dynamic_over_triangle,dynamic_output,triangle_output\n");

    for (int idx = 0; idx < EXPERIMENT_IMAGE_COUNT; ++idx) {
        char image_path[512];
        snprintf(image_path, sizeof(image_path), "%s/%s", images_dir, EXPERIMENT_IMAGE_FILES[idx]);

        int w = 0;
        int h = 0;
        int cpp = 0;
        unsigned char *loaded = stbi_load(image_path, &w, &h, &cpp, COLOR_CHANNELS);

        int seams_for_image = seam_number;
        if (seams_for_image > w - 1) seams_for_image = w - 1;


        // Averaging variables
        double dyn_energy_sum = 0.0, dyn_seam_sum = 0.0, dyn_remove_sum = 0.0, dyn_total_sum = 0.0;
        double tri_energy_sum = 0.0, tri_seam_sum = 0.0, tri_remove_sum = 0.0, tri_total_sum = 0.0;

        for (int rep = 0; rep < BENCHMARK_REPEATS; ++rep) {
            timing_result_t dynamic_result, triangle_result;
            unsigned char *dyn_out = NULL, *tri_out = NULL;
            run_dynamic_comparison_once(loaded, w, h, cpp, seams_for_image, num_threads, num_threads, num_threads, &dyn_out, &dynamic_result);
            run_triangle_comparison_once(loaded, w, h, cpp, seams_for_image, num_threads, num_threads, num_threads, strip_height, &tri_out, &triangle_result);
            dyn_energy_sum += dynamic_result.energy;
            dyn_seam_sum += dynamic_result.seam;
            dyn_remove_sum += dynamic_result.remove;
            dyn_total_sum += dynamic_result.total;
            tri_energy_sum += triangle_result.energy;
            tri_seam_sum += triangle_result.seam;
            tri_remove_sum += triangle_result.remove;
            tri_total_sum += triangle_result.total;
        }

        double dyn_energy_avg = dyn_energy_sum / BENCHMARK_REPEATS;
        double dyn_seam_avg = dyn_seam_sum / BENCHMARK_REPEATS;
        double dyn_remove_avg = dyn_remove_sum / BENCHMARK_REPEATS;
        double dyn_total_avg = dyn_total_sum / BENCHMARK_REPEATS;
        double tri_energy_avg = tri_energy_sum / BENCHMARK_REPEATS;
        double tri_seam_avg = tri_seam_sum / BENCHMARK_REPEATS;
        double tri_remove_avg = tri_remove_sum / BENCHMARK_REPEATS;
        double tri_total_avg = tri_total_sum / BENCHMARK_REPEATS;

        double total_speedup = (tri_total_avg > 0.0) ? (dyn_total_avg / tri_total_avg) : 0.0;
        double seam_speedup = (tri_seam_avg > 0.0) ? (dyn_seam_avg / tri_seam_avg) : 0.0;

        char base_name[256];
        snprintf(base_name, sizeof(base_name), "%s", EXPERIMENT_IMAGE_FILES[idx]);
        char *dot = strrchr(base_name, '.');
        if (dot) *dot = '\0';

        char dynamic_output_file[256];
        char triangle_output_file[256];
        snprintf(dynamic_output_file, sizeof(dynamic_output_file), "%s/%s_dynamic.png", results_dir, base_name);
        snprintf(triangle_output_file, sizeof(triangle_output_file), "%s/%s_triangle.png", results_dir, base_name);

        fprintf(csv,
                "%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f\n",
                EXPERIMENT_IMAGE_FILES[idx],
                w,
                h,
                cpp,
                seams_for_image,
                strip_height,
                dyn_energy_avg,
                dyn_seam_avg,
                dyn_remove_avg,
                dyn_total_avg,
                tri_energy_avg,
                tri_seam_avg,
                tri_remove_avg,
                tri_total_avg,
                total_speedup,
                seam_speedup);

        stbi_image_free(loaded);
    }

    fclose(csv);
    return 1;
}

static int run_greedy_vs_dynamic_experiment(const char *images_dir, int seam_number, int batch_size, int num_threads, const char *csv_path)
{
    FILE *csv = fopen(csv_path, "w");
    char results_dir[256];
    snprintf(results_dir, sizeof(results_dir), "greedy_vs_dynamic_results");

    system("mkdir -p greedy_vs_dynamic_results");

    fprintf(csv, "image,width,height,channels,seams,greedy_runtime_s,dynamic_runtime_s,speedup,greedy_output,dynamic_output,visual_notes\n");

    for (int idx = 0; idx < EXPERIMENT_IMAGE_COUNT; ++idx) {
        char image_path[512];
        snprintf(image_path, sizeof(image_path), "%s/%s", images_dir, EXPERIMENT_IMAGE_FILES[idx]);

        int w = 0, h = 0, cpp = 0;
        unsigned char *loaded = stbi_load(image_path, &w, &h, &cpp, COLOR_CHANNELS);

        int seams_for_image = seam_number;
        if (seams_for_image > w - 1) seams_for_image = w - 1;


        // Averaging variables
        double greedy_runtime_sum = 0.0, dynamic_total_sum = 0.0;

        for (int rep = 0; rep < BENCHMARK_REPEATS; ++rep) {
            greedy_vs_dynamic_result_t greedy_result_tmp;
            timing_result_t dynamic_result_tmp;
            unsigned char *greedy_out = NULL, *dynamic_out = NULL;
            run_greedy_benchmark_once(loaded, w, h, cpp, seams_for_image, batch_size, num_threads, &greedy_out, &greedy_result_tmp);
            run_dynamic_comparison_once(loaded, w, h, cpp, seams_for_image, num_threads, num_threads, num_threads, &dynamic_out, &dynamic_result_tmp);
            greedy_runtime_sum += greedy_result_tmp.runtime;
            dynamic_total_sum += dynamic_result_tmp.total;
        }

        
        double greedy_runtime_avg = greedy_runtime_sum / BENCHMARK_REPEATS;
        double dynamic_total_avg = dynamic_total_sum / BENCHMARK_REPEATS;
        double speedup = dynamic_total_avg / greedy_runtime_avg;
        
        fprintf(csv, "%s,%d,%d,%d,%d,%.6f,%.6f,%.2f\n", EXPERIMENT_IMAGE_FILES[idx], w, h, cpp, seams_for_image, greedy_runtime_avg, dynamic_total_avg, speedup);

        stbi_image_free(loaded);
    }

    fclose(csv);
    return 1;
}

static int run_carving_many_times (const char *images_dir, int seam_number, int num_threads, const char *csv_path)
{
    FILE *csv = fopen(csv_path, "w");
    if (csv == NULL) {
        return 0;
    }

    timing_result_t global_sum = {0.0, 0.0, 0.0, 0.0};
    int global_runs = 0;

    fprintf(csv, "record,image,iteration,width,height,channels,seams,mode,batch_size,strip_height,energy_s,seam_s,remove_s,total_s\n");

    for (int idx = 0; idx < EXPERIMENT_IMAGE_COUNT; ++idx) {
        char image_path[512];
        snprintf(image_path, sizeof(image_path), "%s/%s", images_dir, EXPERIMENT_IMAGE_FILES[idx]);

        int w = 0, h = 0, cpp = 0;
        unsigned char *loaded = stbi_load(image_path, &w, &h, &cpp, COLOR_CHANNELS);
        if (loaded == NULL) {
            fprintf(csv, "image_load_failed,%s,%d,%d,%d,%d,%d,dynamic,%d,%d,0.000000,0.000000,0.000000,0.000000\n", EXPERIMENT_IMAGE_FILES[idx], 0, 0, 0, 0, seam_number, 0, 0);
            fprintf(csv, "----------------\n");
            continue;
        }

        int seams_for_image = seam_number;
        if (seams_for_image > w - 1) seams_for_image = w - 1;

        timing_result_t image_sum = {0.0, 0.0, 0.0, 0.0};
        int image_runs = 0;

        for (int iter = 0; iter < EXPERIMENT_ITERATION; ++iter) {
            timing_result_t tr;
            if (!run_dynamic_benchmark_once(loaded, w, h, cpp, seams_for_image, num_threads, num_threads, num_threads, &tr)) {
                continue;
            }

            image_sum.energy += tr.energy;
            image_sum.seam += tr.seam;
            image_sum.remove += tr.remove;
            image_sum.total += tr.total;
            image_runs++;

            global_sum.energy += tr.energy;
            global_sum.seam += tr.seam;
            global_sum.remove += tr.remove;
            global_sum.total += tr.total;
            global_runs++;

            fprintf(csv,
                    "iteration,%s,%d,%d,%d,%d,%d,dynamic,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                    EXPERIMENT_IMAGE_FILES[idx],
                    iter + 1,
                    w,
                    h,
                    cpp,
                    seams_for_image,
                    0,
                    0,
                    tr.energy,
                    tr.seam,
                    tr.remove,
                    tr.total);
        }

        timing_result_t image_avg = {0.0, 0.0, 0.0, 0.0};
        if (image_runs > 0) {
            image_avg.energy = image_sum.energy / (double)image_runs;
            image_avg.seam = image_sum.seam / (double)image_runs;
            image_avg.remove = image_sum.remove / (double)image_runs;
            image_avg.total = image_sum.total / (double)image_runs;
        }

        fprintf(csv,
            "image_average,%s,%d,%d,%d,%d,%d,dynamic,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                EXPERIMENT_IMAGE_FILES[idx],
                0,
                w,
                h,
                cpp,
                seams_for_image,
                0,
                0,
                image_avg.energy,
                image_avg.seam,
                image_avg.remove,
                image_avg.total);

        fprintf(csv, "----------------\n");

        stbi_image_free(loaded);
    }

    timing_result_t global_avg = {0.0, 0.0, 0.0, 0.0};
    if (global_runs > 0) {
        global_avg.energy = global_sum.energy / (double)global_runs;
        global_avg.seam = global_sum.seam / (double)global_runs;
        global_avg.remove = global_sum.remove / (double)global_runs;
        global_avg.total = global_sum.total / (double)global_runs;
    }

    fprintf(csv, "overall_average,ALL_IMAGES,%d,%d,%d,%d,%d,dynamic,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
            0,
            0,
            0,
            0,
            seam_number,
            0,
            0,
            global_avg.energy,
            global_avg.seam,
            global_avg.remove,
            global_avg.total);

    fclose(csv);
    return 1;
}

int main(void)
{
    const char *images_dir = "../test_images";
    const int seam_number = 32;
    const int thread_max = omp_get_num_procs();
    const int strip_height = 32;
    const int batch_size = 32;

    // Dynamic thread experiment (optional)
    // run_dynamic_thread_experiment(images_dir, 128, 1, thread_max, 1, "dynamic_thread_experiment.csv");

    // Greedy vs Dynamic comparison experiment
    // const char *comparison_csv = "greedy_vs_dynamic_comparison.csv";
    // const int batch_size = 16;
    // run_greedy_vs_dynamic_experiment(images_dir, 512, batch_size, thread_max, "greedy_vs_dynamic_comparison_avg.csv");

    // Dynamic vs improved triangle
    //run_dynamic_vs_triangle_experiment(images_dir, 512, thread_max, strip_height, "triangle_vs_dynamic_comparison.csv");

    // Base benchmark of running dynamic many times
    run_carving_many_times(images_dir, 128, 2, "carving_many_times-2thread.csv");
    run_carving_many_times(images_dir, 128, 4, "carving_many_times-4thread.csv");
    run_carving_many_times(images_dir, 128, 8, "carving_many_times-8thread.csv");
    run_carving_many_times(images_dir, 128, 16, "carving_many_times-16thread.csv");
    run_carving_many_times(images_dir, 128, 32, "carving_many_times-32thread.csv");
    run_carving_many_times(images_dir, 128, 64, "carving_many_times-64thread.csv");
    // modified to use non optimised energy calculation
    // run_carving_many_times(images_dir, 128, thread_max, "carving_many_times-Naive-Energy.csv");
    // Compiled with -O2 and no OpenMP
    // run_carving_many_times(images_dir, 128, 1, "carving_many_times-Sequential.csv");

    return 0;
}
