#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "carving.h"
#include "stb_image.h"

#define COLOR_CHANNELS 0

#define EXPERIMENT_IMAGES_DIR "../test_images"
#define EXPERIMENT_OUTPUT_CSV "dynamic_thread_experiment.csv"
#define EXPERIMENT_SEAM_NUMBER 32
#define EXPERIMENT_THREAD_MIN 1
#define EXPERIMENT_THREAD_STEP 1

static const char *EXPERIMENT_IMAGE_FILES[] = {
    "720x480.png",
    "1024x768.png",
    "1920x1200.png",
    "3840x2160.png",
    "7680x4320.png"
};

#define EXPERIMENT_IMAGE_COUNT 5

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

        printf("Image: %s | size=%dx%d cpp=%d | seams=%d\n", EXPERIMENT_IMAGE_FILES[idx], w, h, cpp, seams_for_image);

        double best_energy = 1e300;
        double best_seam = 1e300;
        double best_remove = 1e300;
        double best_total = 1e300;
        int best_energy_threads = thread_min;
        int best_seam_threads = thread_min;
        int best_remove_threads = thread_min;

        const int base_threads = thread_max;

        for (int threads = thread_min; threads <= thread_max; threads += thread_step) {
            timing_result_t tr;
            run_dynamic_benchmark_once(loaded, w, h, cpp, seams_for_image, threads, base_threads, base_threads, &tr);
            printf("  energy sweep: e=%d s=%d r=%d | energy=%.6f s seam=%.6f s remove=%.6f s total=%.6f s\n", threads, base_threads, base_threads, tr.energy, tr.seam, tr.remove, tr.total);
            if (tr.energy < best_energy) {
                best_energy = tr.energy;
                best_energy_threads = threads;
            }
        }

        for (int threads = thread_min; threads <= thread_max; threads += thread_step) {
            timing_result_t tr;
            run_dynamic_benchmark_once(loaded, w, h, cpp, seams_for_image, base_threads, threads, base_threads, &tr);
            printf("  seam sweep:   e=%d s=%d r=%d | energy=%.6f s seam=%.6f s remove=%.6f s total=%.6f s\n", base_threads, threads, base_threads, tr.energy, tr.seam, tr.remove, tr.total);
            if (tr.seam < best_seam) {
                best_seam = tr.seam;
                best_seam_threads = threads;
            }
        }

        for (int threads = thread_min; threads <= thread_max; threads += thread_step) {
            timing_result_t tr;
            run_dynamic_benchmark_once(loaded, w, h, cpp, seams_for_image, base_threads, base_threads, threads, &tr);
            printf("  remove sweep: e=%d s=%d r=%d | energy=%.6f s seam=%.6f s remove=%.6f s total=%.6f s\n", base_threads, base_threads, threads, tr.energy, tr.seam, tr.remove, tr.total);
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

        printf("  best -> energy:%d seam:%d remove:%d | combined total=%.6f s\n\n", best_energy_threads, best_seam_threads, best_remove_threads, best_total);
        stbi_image_free(loaded);
    }

    fclose(csv);
    return 1;
}

int main(void)
{
    const char *images_dir = EXPERIMENT_IMAGES_DIR;
    const char *csv_path = EXPERIMENT_OUTPUT_CSV;
    const int seam_number = EXPERIMENT_SEAM_NUMBER;
    const int thread_min = EXPERIMENT_THREAD_MIN;
    const int thread_max = omp_get_num_procs();
    const int thread_step = EXPERIMENT_THREAD_STEP;

    if (!run_dynamic_thread_experiment(images_dir, seam_number, thread_min, thread_max, thread_step, csv_path)) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
