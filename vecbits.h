#ifndef VECBITS_H
#define VECBITS_H

#include <stdbool.h>

typedef struct {
    size_t total_weight; /* somatorio de pesos */
    size_t total_value; /* somatorio de valor */
    bool *bits;
} VecBits;

VecBits vec_bits(size_t t, size_t w, Item *items);
size_t vec_bits_value(size_t t, VecBits *m, Item *items);
size_t vec_bits_weight(size_t t, VecBits *m, Item *items);
void vec_bits_show(size_t t, VecBits *m);

#ifdef VECBITS_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>

/*
 inicializa uma mochila em um vetor de bits
 t = total de itens
 bits = vetor de bits
*/
VecBits vec_bits(size_t t, size_t w, Item *items) {
    assert(items != NULL);

    VecBits m = {
        .total_value = 0,
        .total_weight = 0,
        .bits = NULL,
    };
    
    m.bits = malloc(t * sizeof(bool));
    assert(m.bits != NULL);

    for (size_t i = 0; i < t; ++i) {
        m.bits[i] = rand() % 2;
        size_t next_weight = m.total_weight + items[i].weight;

        if (m.bits[i] && next_weight < w) {
            m.total_value = m.total_value + items[i].value;
            m.total_weight = next_weight;
        }
        
        if (m.bits[i] && next_weight > w) m.bits[i] = false;
    }
    return m;
}

size_t vec_bits_value(size_t t, VecBits *m, Item *items) {
    assert(m != NULL);
    assert(items != NULL);
    
    size_t total = 0;
    for (size_t i = 0; i < t; ++i) {
        if (m->bits[i]) {
            total += items[i].value;
        }
    }
    return total;
}

size_t vec_bits_weight(size_t t, VecBits *m, Item *items) {
    assert(m != NULL);
    assert(items != NULL);
    
    size_t total = 0;
    for (size_t i = 0; i < t; ++i) {
        if (m->bits[i]) {
            total += items[i].weight;
        }
    }
    return total;
}

void vec_bits_show(size_t t, VecBits *m) {
    assert(m != NULL);
    assert(m->bits != NULL);
    printf("vetor [");
    
    for (size_t i = 0; i < t; ++i) {
        printf("%2d", m->bits[i]);
    }
    printf(" ] (valor = %d peso = %d)\n", m->total_value, m->total_weight);
}

#endif /* VECBITS_IMPLEMENTATION */

#endif /* VECBITS_H */
