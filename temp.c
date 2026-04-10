#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <math.h>

// capacidade da mochila
#define W 200
// minimo peso de um item
#define MIN_W 1
// maximo valor de um item 
#define V 100
// minimo valor de um item
#define MIN_V 1
// taxa de penalizacao, 10 por item
#define R 10 
// total items
#define T 15

#define ITEM_IMPLEMENTATION
#include "item.h"

#define VECBITS_IMPLEMENTATION
#include "vecbits.h"

typedef struct {
    size_t next_state; /* index que indica o estado escolhido para proximo */
    size_t total_states; /* tamanho do espaco de estados vizinhos */
    size_t vec_size; /* tamanho de um estado */
    VecBits state; /* estado atual */
    VecBits nstates; /* vizinhanca de um estado */
} NeighborState;


/*
  cria o espaco de vizinhos que representado por o estado atual mais a posicao que varia o bit
  ns->state = [b0, b1, ..., bn];
  ns->nstates = [!b0, !b1, ..., !bn];

  novo estado = ns->state[b0, b1, ns->nstate[i], bn];
 */
void neighbor_space(NeighborState *ns) {
    assert(ns != NULL);
    assert(ns->state.bits != NULL);
    assert(ns->nstates.bits != NULL);

    /* para cada estado do estado atual associa um posicao ao espaco de estados variando 1 item */
    for (size_t i = 0; i < ns->total_states; ++i) {
        ns->nstates.bits[i] = !ns->state.bits[i];
    }
}

/*
  calcula proximo estado com base no estado atual, sorteia um numero e faz o flip
  s = [0 1 0]
  n = [1 0 1]

  vizinhos = [ 1 1 0 ]
           = [ 0 0 0 ]
           = [ 0 1 1 ]
  
 */

void neighbor_space_next_state(Item *items, NeighborState *ns) {
    assert(ns != NULL);
    assert(ns->state.bits != NULL);
    assert(ns->nstates.bits != NULL);
    size_t i = rand() % ns->total_states;
    ns->next_state = i;

    ns->nstates.total_value = ns->state.total_value;
    ns->nstates.total_weight = ns->state.total_weight;
        
    if (!ns->state.bits[i]) {
        ns->nstates.total_value += items[i].value;
        ns->nstates.total_weight += items[i].weight;
    } else {
        ns->nstates.total_value -= items[i].value;
        ns->nstates.total_weight -= items[i].weight;
    }
}

void neighbor_space_next_state_show(NeighborState *ns) {
    assert(ns != NULL);
    assert(ns->state.bits != NULL);
    assert(ns->nstates.bits != NULL);

    printf("proximo estado: vetor [");
    for (size_t i = 0; i < ns->vec_size; ++i) {
        if (ns->next_state == i) {
            printf("%2d", !(ns->state.bits[i]));
        } else {
            printf("%2d", ns->state.bits[i]);
        }
    }
    
    printf(" ] (valor = %d peso = %d)\n", ns->nstates.total_value, ns->nstates.total_weight);
}

void neighbor_space_show(Item *items, NeighborState *ns) {
    assert(ns != NULL);
    assert(ns->state.bits != NULL);
    assert(ns->nstates.bits != NULL);
    printf("estado inicial: ");
    vec_bits_show(ns->vec_size, &ns->state);
    neighbor_space_next_state_show(ns);
    printf("estados vizinhos: \n");

    size_t weight = ns->state.total_weight;
    size_t value = ns->state.total_value;
    for (size_t i = 0; i < ns->total_states; ++i) {
        printf("vetor [");
        
        for (size_t j = 0; j < ns->vec_size; ++j) {
            if (j == i) {
                printf("%2d", !(ns->state.bits[j]));

                if (!ns->state.bits[j]) {
                    weight += items[i].weight;
                    value += items[i].value;
                } else {
                    weight -= items[i].weight;
                    value -= items[i].value;
                }
            } else {
                printf("%2d", ns->state.bits[j]);
            }
        }
        
        printf(" ] (valor = %d peso = %d)\n", value, weight);
        weight = ns->state.total_weight;
        value =  ns->state.total_value;
    }
}


int neighbor_space_evaluate(size_t w, size_t rate, size_t total_weight, size_t total_value) {
    if (total_weight <= w) {
        return total_value;
    } else {
        size_t ex = total_weight - w;
        size_t factor = ex * rate;
        return (int)total_value - (int)factor;
    }
}

void tempering_annelling(size_t t, size_t w, size_t r, size_t max_temperature, Item *items, NeighborState *ns) {

    neighbor_space(ns);
    neighbor_space_next_state(items, ns);
    neighbor_space_show(items, ns);

    size_t tempo = 100;
    size_t max_steps = t * tempo;
    
    double temp = max_temperature;
    double alpha = temp / max_steps;

    printf("tempera simulada\n");
    printf("passos totais: %d temperatura inicial: %.2f taxa alpha: %f\n", max_steps, temp, alpha);

    VecBits best_state = vec_bits_empty(t);
    memcpy(best_state.bits, ns->state.bits, t * sizeof(bool));
    
    best_state.total_value = ns->state.total_value;
    best_state.total_weight = ns->state.total_weight;

    for (size_t step = 0; step < max_steps; ++step) {

        vec_bits_show(t, &best_state);
        if (temp <= 0.001) break;

        neighbor_space_next_state(items, ns);

        int current_state = neighbor_space_evaluate(w, r, ns->state.total_weight, ns->state.total_value);
        int next_state = neighbor_space_evaluate(w, r, ns->nstates.total_weight, ns->nstates.total_value);

        int delta_e = next_state - current_state;
        bool action = false;

        /* decide se vai ir pro proximo estado */
        if (delta_e > 0) {
            action = true; 
        } else {
            double bad_prob = exp((double)delta_e / temp);
            double rand_prob = (double)rand() / (double)RAND_MAX;

            /* probabilidade de pegar caminho pior*/
            if (rand_prob < bad_prob) {
                action = true;
            }
        }

        if (action) {
            size_t idx_next = ns->next_state;
            ns->state.bits[idx_next] = !ns->state.bits[idx_next];
            
            ns->state.total_value = ns->nstates.total_value;
            ns->state.total_weight = ns->nstates.total_weight;

            if (next_state > (int)best_state.total_value && ns->state.total_weight <= w) {
                memcpy(best_state.bits, ns->state.bits, t * sizeof(bool));
                best_state.total_value = ns->state.total_value;
                best_state.total_weight = ns->state.total_weight;
            }
        }
        temp -= alpha;
    }

    printf("melhor estado\n");
    vec_bits_show(t, &best_state);

    free(best_state.bits);
    free(ns->state.bits);
    free(ns->nstates.bits);    
}

int main(void) {
    srand(time(NULL));
    
    Item items[T];
    init_random_items(T, items);
    items_show(T, items);

    NeighborState ns = (NeighborState){
        .next_state = 0,
        .total_states = T,
        .vec_size = T,
        .state = vec_bits(T, W, items),
        .nstates = vec_bits_empty(T),
    };
    tempering_annelling(T, W, R, V, items, &ns);
    items_show(T, items);
    return 0;
}

