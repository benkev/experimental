#include <stdio.h>
#include <locale.h>

int main() {
    
    setlocale(LC_NUMERIC, "");
    printf("%10d\n", 1123456789);
    printf("%'10d\n", 1123456789);

    return 0;
}

