#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>

// total de geracoes consecutivas
#define N 20
// total de geracoes
#define G 100
// capacidade da mochila
#define W 200
// minimo peso de um item
#define MIN_W 1
// total items
#define T 15
// maximo valor de um item 
#define V 100
// minimo valor de um item
#define MIN_V 1
// total de individuos
#define I 10
// total de individuos do torneio
#define K 2
// porcentagem de mutacao
#define M_P 0.05

#define ITEM_IMPLEMENATION
#include "item.h"

#define VECBITS_IMPLEMENTATION
#include "vecbits.h"

typedef struct {
    size_t size; /* tamanho da populacao */
    size_t vec_size; /* tamanho de um individuo */
    VecBits *vecsbits; 
} Population;


size_t fitness(size_t w, VecBits *m) {
    assert(m != NULL);
    if (m->total_weight > w) return 0;
    return m->total_value;
}

void vec_bits_crossover(size_t t, VecBits *v1, VecBits *v2, VecBits *f1, VecBits *f2) {
    assert(v1 != NULL);
    assert(v2 != NULL);
    assert(f1 != NULL);
    assert(f2 != NULL);
    
    size_t p1 = rand() % t;
    size_t p2 = p1 + 1 + (rand() % (t - p1));
    /* printf("(%d, %d)\n", p1, p2); */

    if (p1 > 0) {
        memcpy(&f1->bits[0], &v1->bits[0], p1 * sizeof(bool));
        memcpy(&f2->bits[0], &v2->bits[0], p1 * sizeof(bool));
    }
    
    memcpy(&f1->bits[p1], &v2->bits[p1], (p2 - p1) * sizeof(bool));
    memcpy(&f2->bits[p1], &v1->bits[p1], (p2 - p1) * sizeof(bool));

    if (p2 < t) {
        memcpy(&f1->bits[p2], &v1->bits[p2], (t - p2) * sizeof(bool));
        memcpy(&f2->bits[p2], &v2->bits[p2], (t - p2) * sizeof(bool));
    }
}

void vec_bits_mutation(size_t t, double mp, VecBits *m) {
    assert(m != NULL);
    for (size_t i = 0; i < t; ++i) {
        if (((double) rand() / (double) RAND_MAX) <= mp) {
            m->bits[i] = !m->bits[i];
        }
    }
}

void init_random_items(size_t t, Item *items) {
    assert(items != NULL);

    for (size_t i = 0; i < t; ++i) {
        items[i] = (Item){
            .value = MIN_V + (rand() % (V - MIN_V + 1)),
            .weight = MIN_W + (rand() % (W - MIN_W + 1)),
        };
    }
}

void items_show(size_t t, Item *items) {
    assert(items != NULL);
    printf("itens[%zu] { ", t);
    for (size_t i = 0; i < t; ++i) {
        printf("(%d, %d) ", items[i].value, items[i].weight);
    }
    printf("} W = %d\n", W);
}


/*
  cria uma populacao inicial randomica
*/
void population_start(size_t w, Population *pop, Item *items) {
    assert(pop != NULL);
    assert(items != NULL);
    
    pop->vecsbits = malloc(pop->size * sizeof(VecBits));
    assert(pop->vecsbits != NULL);
    
    for (size_t i = 0; i < pop->size; ++i) {
        pop->vecsbits[i] = vec_bits(pop->vec_size, w, items);
    }
}

void population_show(Population *pop) {
    assert(pop != NULL);
    assert(pop->vecsbits != NULL);

    for (size_t i = 0; i < pop->size; ++i) {
        vec_bits_show(pop->vec_size, &pop->vecsbits[i]);
    }    
}

void population_free(Population *pop) {
    assert(pop != NULL);
    assert(pop->vecsbits != NULL);

    for (size_t i = 0; i < pop->size; ++i) {
        free(pop->vecsbits[i].bits);
    }
    free(pop->vecsbits);
}

/*
  faz o torneio na populacao de k vetores,
  seleciona um vetor com melhor fitness
*/
size_t population_tournament(size_t w, size_t k, Population *pop) {
    assert(pop != NULL);
    size_t kvec[k];
    /* printf("torneio ["); */
    for (size_t i = 0; i < k; ++i) {
        size_t idx = rand() % pop->size;
        kvec[i] = idx;
        /* printf(" %d", idx); */
    }
    /* printf(" ]");  */

    size_t selected = kvec[0];
    size_t fit = fitness(w, &pop->vecsbits[kvec[0]]);
    for (size_t i = 1; i < k; ++i) {
        size_t current_fit = fitness(w, &pop->vecsbits[kvec[i]]);
        if (fit < current_fit) {
            selected = kvec[i];
            fit = current_fit;
        }
    }
    /* printf(" (vencedor = %d)\n", selected); */
    
    return selected;
}

/*
  executa o torneio, e cria populacao para crossover
*/
void population_selection(size_t w, size_t k, size_t *selections, Population *pop) {
    assert(pop != NULL);
    assert(selections != NULL);

    for (size_t i = 0; i < pop->size; ++i) {
        selections[i] = population_tournament(w, k, pop);
    }
}

void population_crossover(size_t *selections, Population *ipop, Population *opop) {
    assert(selections != NULL);
    assert(ipop != NULL);
    assert(opop != NULL);
    assert(opop->vecsbits != NULL);
   
    for (size_t i = ipop->size; i > 1; i -= 2) {
         VecBits *v1 = &ipop->vecsbits[selections[i-1]];
         VecBits *v2 = &ipop->vecsbits[selections[i-2]];
         VecBits *f1 = &opop->vecsbits[i-1];
         VecBits *f2 = &opop->vecsbits[i-2];
    
         vec_bits_crossover(ipop->vec_size, v1, v2, f1, f2);       
    }
}

void population_mutation(double mp, Item *items, Population *pop) {
    assert(pop != NULL);
    for (size_t i = 0; i < pop->size; ++i) {
        vec_bits_mutation(pop->vec_size, mp, &pop->vecsbits[i]);
        pop->vecsbits[i].total_value = vec_bits_value(pop->vec_size, &pop->vecsbits[i], items);
        pop->vecsbits[i].total_weight = vec_bits_weight(pop->vec_size, &pop->vecsbits[i], items);
    }
}

size_t population_best_fitness(size_t w, Population *pop) {
    size_t index = 0;

    size_t best_fit = fitness(w, &pop->vecsbits[index]); 

    for (size_t i = 1; i < pop->size; ++i) {
        size_t current_fit = fitness(w, &pop->vecsbits[i]);
        if (best_fit < current_fit) {
            index = i;
            best_fit = current_fit;
        }
    }
    return index;
}

void population_elitism(size_t best_fit, size_t w, Item *items, Population *ipop, Population *opop) {
    assert(items != NULL);
    assert(ipop != NULL);
    assert(opop != NULL);

    size_t index = 0;
    size_t worst_fit = fitness(w, &opop->vecsbits[index]);
    for (size_t i = 1; i < opop->size; ++i) {
        size_t current_fit = fitness(w, &opop->vecsbits[i]);
        if (worst_fit > current_fit) {
            index = i;
            worst_fit = current_fit;
        }
    }

    opop->vecsbits[index].total_value = ipop->vecsbits[best_fit].total_value;
    opop->vecsbits[index].total_weight = ipop->vecsbits[best_fit].total_weight;
    memcpy(opop->vecsbits[index].bits, ipop->vecsbits[best_fit].bits, opop->vec_size * sizeof(bool));
}

void population_survivor(size_t w, size_t k, size_t g, size_t n, double mp, Item *items, Population *pop) {
    assert(items != NULL);
    assert(pop != NULL);
    
    size_t gen_counter = 0;
    size_t stagnation_counter = 0;

    size_t best_idx = population_best_fitness(w, pop);
    size_t best_val = fitness(w, &pop->vecsbits[best_idx]);

    Population next_pop = {
        .size = pop->size,
        .vec_size = pop->vec_size,
        .vecsbits = malloc(pop->size * sizeof(VecBits))
    };
    assert(next_pop.vecsbits != NULL);

    for (size_t i = 0; i < next_pop.size; ++i) {
        next_pop.vecsbits[i].bits = malloc(next_pop.vec_size * sizeof(bool));
        assert(next_pop.vecsbits[i].bits != NULL);
    }

    size_t *selections = malloc(pop->size * sizeof(size_t));
    assert(selections != NULL);
    
    while (gen_counter < g && stagnation_counter < n) {
        gen_counter += 1;
        population_show(pop);
        printf("geracao: %zu nivel de estagnacao: %zu\n", gen_counter, stagnation_counter);
        items_show(pop->vec_size, items);
            
        population_selection(w, k, selections, pop);
        population_crossover(selections, pop, &next_pop);
        population_mutation(mp, items, &next_pop);
        population_elitism(best_idx, w, items, pop, &next_pop);

        VecBits *temp_vecs = pop->vecsbits;
        pop->vecsbits = next_pop.vecsbits;
        next_pop.vecsbits = temp_vecs;

        best_idx = population_best_fitness(w, pop);
        size_t current_best_val = fitness(w, &pop->vecsbits[best_idx]);

        if (current_best_val > best_val) {
            best_val = current_best_val;
            stagnation_counter = 0;
        } else {
            stagnation_counter += 1;
        }
    }

    size_t final_best_idx = population_best_fitness(w, pop);
    printf("best fitness: ");
    vec_bits_show(pop->vec_size, &pop->vecsbits[final_best_idx]);
    
    free(selections);
    population_free(&next_pop);
}

int main(void) {
    srand(time(NULL));
    
    Item items[T];
    init_random_items(T, items);
    items_show(T, items);
    Population pop = {
        .size = I,
        .vec_size = T,
        .vecsbits = NULL,
    };

    population_start(W, &pop, items);
    population_survivor(W, K, G, N, M_P, items, &pop);
    population_free(&pop);
    
    /* VecBits m = vec_bits(T, items); */
    /* vec_bits_show(T, &m); */
    return 0;
}
