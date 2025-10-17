#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>

int main() {
    printf("Testing network connectivity...\n");
    
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket() failed");
        return 1;
    }
    printf("Socket created successfully\n");
    
    struct sockaddr_in dest;
    memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;
    dest.sin_port = htons(80);
    inet_pton(AF_INET, "89.187.177.114", &dest.sin_addr);
    
    char buffer[100] = "TEST";
    ssize_t sent = sendto(sock, buffer, strlen(buffer), 0, 
                         (struct sockaddr*)&dest, sizeof(dest));
    
    if (sent < 0) {
        printf("sendto() failed: %s\n", strerror(errno));
        printf("Error code: %d\n", errno);
    } else {
        printf("Successfully sent %zd bytes to target!\n", sent);
    }
    
    close(sock);
    return 0;
}
