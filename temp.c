#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>

// capacidade da mochila
#define W 200
// minimo peso de um item
#define MIN_W
// maximo valor de um item 
#define V 100
// minimo valor de um item
#define MIN_V 1

// total items
#define T 15

#define ITEM_IMPLEMENTATION
#include "item.h"

#define VECBITS_IMPLEMENTATION
#include "vecbits.h"

int main(void) {
    srand(time(NULL));
    
    Item items[T];
    init_random_items(T, items);
    items_show(T, items);    
    VecBits m = vec_bits(T, W, items);
    vec_bits_show(T, &m);
    return 0;
}

