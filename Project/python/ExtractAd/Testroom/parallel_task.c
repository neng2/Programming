#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 4
#define LIMIT 1000

typedef struct {
    long thread_id;
    unsigned long long result;
} ThreadData;

void* cpu_bound_task(void* arg) {
    ThreadData *data = (ThreadData *)arg;
    unsigned long long sum = 0;
    for (int i = 0; i < LIMIT; i++) {
        sum += i * i;
    }
    data->result = sum;
    printf("Thread %ld: Result = %llu\n", data->thread_id, sum);
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    for (long i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        pthread_create(&threads[i], NULL, cpu_bound_task, (void *)&thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    unsigned long long total = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        total += thread_data[i].result;
    }
    printf("Total sum = %llu\n", total);

    return 0;
}