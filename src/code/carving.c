#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <numa.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

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
// uses channel ch (0=R). You can switch to luminance later.
void calculate_energy(unsigned char *out, const unsigned char *image_in, int w, int h, int cpp)
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
    }

    const int channels_to_use = (cpp >= 3) ? 3 : 1;

    #define CLP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
    #define PIX(ii, jj, ch) image_in[((ii) * w + (jj)) * cpp + (ch)]

    // Helper writes one grayscale energy value into RGB (or single channel for grayscale).
    #define STORE_ENERGY(ii, jj, evalue)                \
        do {                                            \
            int o = ((ii) * w + (jj)) * cpp;            \
            unsigned char ev = (unsigned char)(evalue); \
            image_out[o + 0] = ev;                            \
            if (cpp > 1) image_out[o + 1] = ev;               \
            if (cpp > 2) image_out[o + 2] = ev;               \
            if (cpp > 3) image_out[o + 3] = image_in[o + 3];        \
        } while (0)

    // 1) Interior pixels: no clamp checks needed because all neighbors are in-bounds.
    // i = row index, j = column index.
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < h - 1; ++i) {
        const unsigned char *row_above = in + (i - 1) * w * cpp;
        const unsigned char *row_current = in + i * w * cpp;
        const unsigned char *row_below = in + (i + 1) * w * cpp;

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

            int e = e_sum / channels_to_use;
            if (e > 255) e = 255;
            STORE_ENERGY(i, j, e);
        }
    }

    // 2) First and last columns for every row. Clamp handles neighbors outside [0, w-1].
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

    // 3) First and last rows, excluding corners (corners were already written above).
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

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];

    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int w, h, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &w, &h, &cpp, COLOR_CHANNELS);

    if (image_in == NULL)
    {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d with %d channels.\n", image_in_name, w, h, cpp);
    const size_t datasize = w * h * cpp * sizeof(unsigned char);
    unsigned char *image_out = (unsigned char *)malloc(datasize);
    if (image_out == NULL) {
        printf("Error: Failed to allocate memory for output image!\n");
        stbi_image_free(image_in);
        exit(EXIT_FAILURE);
    }

    // Copy the input image into output and mesure execution time
    // double start = omp_get_wtime();
    // copy_image(image_out, image_in, datasize);
    // double stop = omp_get_wtime();
    // printf("Time to copy: %f s\n", stop - start);
    
    // Calculate the energy/derivatives
    double start = omp_get_wtime();
    calculate_energy(image_out, image_in, w, h, cpp);
    double stop = omp_get_wtime();
    printf("Time to calculate energy: %f s\n", stop - start);

    // Write the output image to file
    char image_out_name_temp[MAX_FILENAME];
    strncpy(image_out_name_temp, image_out_name, MAX_FILENAME);

    const char *file_type = strrchr(image_out_name, '.');
    if (file_type == NULL) {
        printf("Error: No file extension found!\n");
        stbi_image_free(image_in);
        stbi_image_free(image_out);
        exit(EXIT_FAILURE);
    }
    file_type++; // skip the dot

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, w, h, cpp, image_out, w * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, w, h, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, w, h, cpp, image_out);
    else
        printf("Error: Unknown image format %s! Only png, jpg, or bmp supported.\n", file_type);

    // Release the memory
    stbi_image_free(image_in);
    stbi_image_free(image_out);

    return 0;
}