#include <stdio.h>

int main() {
    float sum = 0, sum1 = 0;
    int i = 1;
    while (1) {
        sum1 = sum + 1.0f / (float)i;
        if (sum1 == sum)
            break;
        sum = sum1;
        i++;
        if (i % 100000 == 0) {
            printf("i: %d\n", i);
        }
    }
    printf("Sum: %f, i: %d\n", sum, i);

    return 0;
}