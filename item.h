#ifndef ITEM_H
#define ITEM_H

typedef unsigned char u8;

typedef struct {
    u8 value;
    u8 weight;
} Item;

void init_random_items(size_t t, Item *items);
void items_show(size_t t, Item *items);

#ifdef ITEM_IMPLEMENTATION

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
    printf("itens[%d] { ", t);
    for (size_t i = 0; i < t; ++i) {
        printf("(%d, %d) ", items[i].value, items[i].weight);
    }
    printf("} W = %d\n", W);
}

#endif /* ITEM_IMPLEMENTATION */

#endif /* ITEM_H */
