#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct
{
    int row;
    int col;
    double *elements;
} mat_t;

void m_init(mat_t *_mat, int _row, int _col);
void m_free(mat_t *_mat);
void m_copy(mat_t *_dst, mat_t *_src);
void m_set_elements(mat_t *_mat, double *_arr);
void m_sadd(mat_t *_dst, mat_t *_a, double _b);
void m_smul(mat_t *_dst, mat_t *_a, double _b);
void m_madd(mat_t *_dst, mat_t *_a, mat_t *_b);
void m_msub(mat_t *_dst, mat_t *_a, mat_t *_b);
void m_mmul(mat_t *_dst, mat_t *_a, mat_t *_b);
void m_identify(mat_t *_mat, int _n);
double m_det(mat_t *_mat);
void m_transpose(mat_t *_dst, mat_t *_mat);
void m_inv(mat_t *_dst, mat_t *_mat);
void m_qr(mat_t *_Q, mat_t *_R, mat_t *_mat);
void m_solve(mat_t *_x, mat_t *_mat, mat_t *_b);
void m_eig(mat_t *_mat);
void m_show(mat_t *_mat);