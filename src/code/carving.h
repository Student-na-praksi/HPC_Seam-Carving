#ifndef CARVING_H
#define CARVING_H

#include <stddef.h>

void copy_image(unsigned char *image_out, const unsigned char *image_in, size_t size);
void calculate_energy(unsigned char *image_out, const unsigned char *image_in, int w, int h, int cpp, int rows_per_chunk);
void seam_carving_dynamic(const unsigned char *image_in, int w, int h, int cpp, int rows_per_chunk, int *remove_indexes);
void seam_carving_triangle(const unsigned char *image_in, int w, int h, int cpp, int rows_per_chunk, int *remove_indexes, int strip_height);
void remove_seams(unsigned char *image_out, const unsigned char *image_in, int in_w, int h, int cpp, int rows_per_chunk, const int *remove_indexes);
int seam_carving_greedy(const unsigned char *image_in, int w, int h, int cpp, int rows_per_chunk, int *remove_indexes, int k_remove);
void remove_seams_multi(unsigned char *image_out, const unsigned char *image_in, int in_w, int h, int cpp, int rows_per_chunk, const int *remove_indexes, int seam_count);

#endif
