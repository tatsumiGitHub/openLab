#include "matrix.h"

void m_init(mat_t *_mat, int _row, int _col)
{
    int i, size = _row * _col;
    double *tmp = (double *)malloc(sizeof(double) * size);
    if (tmp == NULL)
    {
        return;
    }
    for (i = 0; i < size; i++)
    {
        tmp[i] = 0.0;
    }
    _mat->elements = tmp;
    _mat->row = _row;
    _mat->col = _col;
}

void m_free(mat_t *_mat)
{
    _mat->row = 0;
    _mat->col = 0;
    free(_mat->elements);
    _mat->elements = NULL;
}

void m_copy(mat_t *_dst, mat_t *_src)
{
    m_free(_dst);
    _dst->row = _src->row;
    _dst->col = _src->col;
    if ((_dst->elements = (double *)malloc(sizeof(double) * _dst->row * _dst->col)) == NULL)
    {
        return;
    }
    memcpy(_dst->elements, _src->elements, sizeof(double) * _dst->row * _dst->col);
    return;
}

void m_set_elements(mat_t *_mat, double *_arr)
{
    memcpy(_mat->elements, _arr, sizeof(double) * _mat->row * _mat->col);
    return;
}

void m_sadd(mat_t *_dst, mat_t *_a, double _b)
{
    int i, size = _a->row * _a->col;
    mat_t tmp = {0, 0, NULL};
    m_init(&tmp, _a->row, _a->col);
    for (i = 0; i < size; i++)
    {
        tmp.elements[i] = _a->elements[i] + _b;
    }
    m_copy(_dst, &tmp);
    m_free(&tmp);
    return;
}

void m_smul(mat_t *_dst, mat_t *_a, double _b)
{
    int i, size = _a->row * _a->col;
    mat_t tmp = {0, 0, NULL};
    m_init(&tmp, _a->row, _a->col);
    for (i = 0; i < size; i++)
    {
        tmp.elements[i] = _a->elements[i] * _b;
    }
    m_copy(_dst, &tmp);
    m_free(&tmp);
    return;
}

void m_madd(mat_t *_dst, mat_t *_a, mat_t *_b)
{
    if (_a->row != _b->row || _a->col != _b->col)
    {
        return;
    }
    int i, size = _a->row * _a->col;
    mat_t tmp = {0, 0, NULL};
    m_init(&tmp, _a->row, _a->col);
    for (i = 0; i < size; i++)
    {
        tmp.elements[i] = _a->elements[i] + _b->elements[i];
    }
    m_copy(_dst, &tmp);
    m_free(&tmp);
    return;
}

void m_msub(mat_t *_dst, mat_t *_a, mat_t *_b)
{
    if (_a->row != _b->row || _a->col != _b->col)
    {
        return;
    }
    int i, size = _a->row * _a->col;
    mat_t tmp = {0, 0, NULL};
    m_init(&tmp, _a->row, _a->col);
    for (i = 0; i < size; i++)
    {
        tmp.elements[i] = _a->elements[i] - _b->elements[i];
    }
    m_copy(_dst, &tmp);
    m_free(&tmp);
    return;
}

void m_mmul(mat_t *_dst, mat_t *_a, mat_t *_b)
{
    if (_a->col != _b->row)
    {
        return;
    }
    int i, j, k;
    mat_t tmp = {0, 0, NULL};
    m_init(&tmp, _a->row, _b->col);
    for (i = 0; i < _a->row; i++)
    {
        for (j = 0; j < _b->col; j++)
        {
            for (k = 0; k < _a->col; k++)
            {
                tmp.elements[i * _b->col + j] += _a->elements[i * _a->col + k] * _b->elements[k * _b->col + j];
            }
        }
    }
    m_copy(_dst, &tmp);
    m_free(&tmp);
    return;
}

void m_hadamard(mat_t *_dst, mat_t *_a, mat_t *_b)
{
    if (_a->row != _b->row || _a->col != _b->col)
    {
        return;
    }
    int i, size = _a->row * _a->col;
    mat_t tmp = {0, 0, NULL};
    m_init(&tmp, _a->row, _a->col);
    for (i = 0; i < size; i++)
    {
        tmp.elements[i] = _a->elements[i] * _b->elements[i];
    }
    m_copy(_dst, &tmp);
    m_free(&tmp);
    return;
}

void m_identify(mat_t *_mat, int _n)
{
    int i, j;
    m_free(_mat);
    _mat->row = _n;
    _mat->col = _n;
    if ((_mat->elements = (double *)malloc(sizeof(double) * _mat->row * _mat->col)) == NULL)
    {
        return;
    }
    for (i = 0; i < _n; i++)
    {
        for (j = 0; j < _n; j++)
        {
            _mat->elements[i * _n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

double m_det(mat_t *_mat)
{
    int i, j, k, n;
    mat_t m = {0, 0, NULL};
    double r, tmp, det = 1;
    if (_mat->row != _mat->col)
    {
        printf("通知：正則行列ではありません．\n");
        return NAN;
    }
    if (_mat->row == 1 && _mat->col == 1)
    {
        return _mat->elements[0];
    }

    n = _mat->row;
    m_init(&m, n, n);
    m_copy(&m, _mat);

    // 上三角行列に変換しつつ、対角成分の積を計算する。
    for (k = 0; k < n - 1; k++)
    {
        if (m.elements[k * n + k] == 0)
        {
            // 対角成分が0だった場合は、その列の値が0でない行と交換する
            for (i = k + 1; i < n; i++)
            {
                if (m.elements[i * n + k] != 0)
                {
                    break;
                }
            }
            if (i < n)
            {
                for (j = 0; j < n; j++)
                {
                    tmp = m.elements[i * n + j];
                    m.elements[i * n + j] = m.elements[k * n + j];
                    m.elements[k * n + j] = tmp;
                }
                det = -det;
            }
        }
        for (i = k + 1; i < n; i++)
        {
            r = m.elements[i * n + k] / m.elements[k * n + k];
            for (j = k; j < n; j++)
            {
                m.elements[i * n + j] -= r * m.elements[k * n + j];
            }
        }
        det *= m.elements[k * n + k];
    }
    det *= m.elements[k * n + k];

    m_free(&m);

    return det;
}

void m_transpose(mat_t *_dst, mat_t *_mat)
{
    int i, j;
    mat_t m = {0, 0, NULL};
    m.row = _mat->col;
    m.col = _mat->row;
    if ((m.elements = (double *)malloc(sizeof(double) * m.row * m.col)) == NULL)
    {
        return;
    }
    for (i = 0; i < m.row; i++)
    {
        for (j = 0; j < m.col; j++)
        {
            m.elements[j * m.col + i] = _mat->elements[i * m.col + j];
        }
    }
    m_copy(_dst, &m);
    m_free(&m);
    return;
}

void m_inv(mat_t *_dst, mat_t *_mat)
{
    if (_mat->row != _mat->col)
    {
        printf("通知：正則行列ではありません．\n");
        return;
    }
    int i, j, k, n = _mat->row;
    double tmp;
    mat_t m = {0, 0, NULL}, inv_m = {0, 0, NULL};

    tmp = m_det(_mat);
    if (fabs(tmp) < __FLT_EPSILON__)
    {
        printf("通知：逆行列が存在しません．\n");
        return;
    }

    m_copy(&m, _mat);
    m_identify(&inv_m, n);

    for (i = 0; i < n; i++)
    {
        tmp = 1 / m.elements[i * n + i];
        for (j = 0; j < n; j++)
        {
            m.elements[i * n + j] *= tmp;
            inv_m.elements[i * n + j] *= tmp;
        }
        for (j = 0; j < n; j++)
        {
            if (i != j)
            {
                tmp = m.elements[j * n + i];
                for (k = 0; k < n; k++)
                {
                    m.elements[j * n + k] -= m.elements[i * n + k] * tmp;
                    inv_m.elements[j * n + k] -= inv_m.elements[i * n + k] * tmp;
                }
            }
        }
    }

    m_copy(_dst, &inv_m);
    m_free(&m);
    m_free(&inv_m);

    return;
}

void m_qr(mat_t *_Q, mat_t *_R, mat_t *_mat)
{
    int i, j, k, m = _mat->row, n = _mat->col;
    double d, c, s;
    double tmp;

    mat_t q_T = {0, 0, NULL}, q = {0, 0, NULL}, r = {0, 0, NULL}, a = {0, 0, NULL};

    m_identify(&q_T, m);
    m_copy(&a, _mat);
    m_copy(&r, &a);

    for (j = 0; j < n; j++)
    {
        for (i = m - 2; j <= i; i--)
        {
            d = sqrt(r.elements[i * n + j] * r.elements[i * n + j] + r.elements[(i + 1) * n + j] * r.elements[(i + 1) * n + j]);
            if (__DBL_EPSILON__ < d)
            {
                c = r.elements[i * n + j] / d;
                s = r.elements[(i + 1) * n + j] / d;
            }
            else
            {
                c = 1.0;
                s = 0.0;
            }
            r.elements[i * n + j] = d;
            r.elements[(i + 1) * n + j] = 0;
            for (k = j + 1; k < n; k++)
            {
                tmp = c * r.elements[i * n + k] + s * r.elements[(i + 1) * n + k];
                r.elements[(i + 1) * n + k] = -s * r.elements[i * n + k] + c * r.elements[(i + 1) * n + k];
                r.elements[i * n + k] = tmp;
            }
            for (k = 0; k < m; k++)
            {
                tmp = c * q_T.elements[i * m + k] + s * q_T.elements[(i + 1) * m + k];
                q_T.elements[(i + 1) * m + k] = -s * q_T.elements[i * m + k] + c * q_T.elements[(i + 1) * m + k];
                q_T.elements[i * m + k] = tmp;
            }
        }
    }

    m_transpose(&q, &q_T);

    m_copy(_Q, &q);
    m_copy(_R, &r);

    m_free(&q_T);
    m_free(&q);
    m_free(&r);
    m_free(&a);
    return;
}

void m_solve(mat_t *_x, mat_t *_mat, mat_t *_b)
{
    int i, j, k, m = _mat->row, n = _mat->col;
    double tmp, d, c, s;

    mat_t r = {0, 0, NULL}, a = {0, 0, NULL}, b = {0, 0, NULL}, x = {0, 0, NULL};

    m_copy(&a, _mat);
    m_copy(&r, &a);
    m_init(&b, m, 1);
    m_init(&x, n, 1);
    if (_b == NULL)
    {
        for (i = 0; i < m; i++)
        {
            b.elements[i] = 1.0;
        }
    }
    else
    {
        m_copy(&b, _b);
    }

    for (j = 0; j < n; j++)
    {
        for (i = m - 2; j <= i; i--)
        {
            d = sqrt(r.elements[i * n + j] * r.elements[i * n + j] + r.elements[(i + 1) * n + j] * r.elements[(i + 1) * n + j]);
            if (__DBL_EPSILON__ < d)
            {
                c = r.elements[i * n + j] / d;
                s = r.elements[(i + 1) * n + j] / d;
            }
            else
            {
                c = 1.0;
                s = 0.0;
            }
            r.elements[i * n + j] = d;
            r.elements[(i + 1) * n + j] = 0;
            for (k = j + 1; k < n; k++)
            {
                tmp = c * r.elements[i * n + k] + s * r.elements[(i + 1) * n + k];
                r.elements[(i + 1) * n + k] = -s * r.elements[i * n + k] + c * r.elements[(i + 1) * n + k];
                r.elements[i * n + k] = tmp;
            }
            tmp = c * b.elements[i] + s * b.elements[i + 1];
            b.elements[i + 1] = -s * b.elements[i] + c * b.elements[i + 1];
            b.elements[i] = tmp;
        }
    }

    x.elements[n - 1] = b.elements[n - 1] / r.elements[(n - 1) * n + n - 1];
    for (i = n - 2; 0 <= i; i--)
    {
        tmp = 0;
        for (j = i; j < n; j++)
        {
            tmp = tmp + r.elements[i * n + j] * x.elements[j];
        }
        x.elements[i] = (b.elements[i] - tmp) / r.elements[i * n + i];
    }

    m_copy(_x, &x);

    m_free(&r);
    m_free(&a);
    m_free(&b);
    m_free(&x);
    return;
}

void m_eig(mat_t *_mat)
{
    if (_mat->row != _mat->col)
    {
        return;
    }
    int i, j, count = 0;
    double eps = 1.0;
    double *eig_val;

    mat_t q = {0, 0, NULL}, r = {0, 0, NULL}, a = {0, 0, NULL}, eig_vec = {0, 0, NULL};

    if ((eig_val = (double *)malloc(sizeof(double) * _mat->row)) == NULL)
    {
        return;
    }
    for (i = 0; i < _mat->row; i++)
    {
        eig_val[i] = 0.0;
    }
    m_copy(&a, _mat);
    m_identify(&eig_vec, _mat->row);
    while (__DBL_EPSILON__ < eps && count < 200)
    {
        m_qr(&q, &r, &a);
        m_mmul(&a, &r, &q);
        m_mmul(&eig_vec, &eig_vec, &q);
        eps = 0.0;
        for (i = 0; i < _mat->row; i++)
        {
            eps += fabs(eig_val[i] - a.elements[i * _mat->row + i]);
            eig_val[i] = a.elements[i * _mat->row + i];
        }
        count++;
    }
    if (count < 200)
    {
        printf("//======== eig ========//\n");
        for (i = 0; i < _mat->row; i++)
        {
            printf("value: %f, ", eig_val[i]);
            printf("vector: [");
            for (j = 0; j < _mat->row; j++)
            {
                printf("%f", eig_vec.elements[i + j * _mat->row]);
                if (j != _mat->row - 1)
                {
                    printf(", ");
                }
                else
                {
                    printf("]\n");
                }
            }
        }
    }
    else
    {
        printf("通知：複素数の固有値を持つ行列（未実装）\n");
    }
    m_free(&q);
    m_free(&r);
    m_free(&a);
    free(eig_val);
    m_free(&eig_vec);

    return;
}

void m_show(mat_t *_mat)
{
    if (_mat->row == 0 || _mat->col == 0)
    {
        return;
    }
    int i, j;

    printf("[");
    for (i = 0; i < _mat->row; i++)
    {
        printf("[");
        for (j = 0; j < _mat->col; j++)
        {
            printf("%f", _mat->elements[i * _mat->col + j]);
            if (j == _mat->col - 1)
            {
                printf("]");
            }
            else
            {
                printf(", ");
            }
        }
        if (i == _mat->row - 1)
        {
            printf("]\n");
        }
        else
        {
            printf(",\n ");
        }
    }
    return;
}
