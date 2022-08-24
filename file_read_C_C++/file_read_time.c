/*
 * Comparison speeds of different methods of file reading. From
 *
 * https://stackoverflow.com/questions/3002122/fastest-file-reading-in-c
 *
 */

/* If you're reading from an actual hard disk, it's going to be slow. The hard disk is your bottle neck, and that's it. */

/* Now, if you're being silly about your call to read/fread/whatever, and say, fread()-ing a byte at a time, then yes, it's going to be slow, as the overhead of fread() will outstrip the overhead of reading from the disk. */

/* If you call read/fread/whatever and request a decent portion of data. This will depend on what you're doing: sometimes all want/need is 4 bytes (to get a uint32), but sometimes you can read in large chunks (4 KiB, 64 KiB, etc. RAM is cheap, go for something significant.) */

/* If you're doing small reads, some of the higher level calls like fread() will actual help you by buffering data behind your back. If you're doing large reads, it might not be helpful, but switching from fread to read will probably not yield that much improvement, as you're bottlenecked on disk speed. */

/* In short: if you can, request a liberal amount when reading, and try to minimize what you write. For large amounts, powers of 2 tend to be friendlier than anything else, but of course, it's OS, hardware, and weather dependent. */

/* So, let's see if this might bring out any differences: */

#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#define BUFFER_SIZE (1 * 1024 * 1024)
#define ITERATIONS (10 * 1024)

double now()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.;
}

int main()
{
    unsigned char buffer[BUFFER_SIZE]; // 1 MiB buffer

    double end_time;
    double total_time;
    int i, x, y;
    double start_time = now();

#ifdef USE_FREAD
    FILE *fp;
    fp = fopen("/dev/zero", "rb");
    for(i = 0; i < ITERATIONS; ++i)
    {
        fread(buffer, BUFFER_SIZE, 1, fp);
        for(x = 0; x < BUFFER_SIZE; x += 1024)
        {
            y += buffer[x];
        }
    }
    fclose(fp);
#elif USE_MMAP
    unsigned char *mmdata;
    int fd = open("/dev/zero", O_RDONLY);
    for(i = 0; i < ITERATIONS; ++i)
    {
        mmdata = mmap(NULL, BUFFER_SIZE, PROT_READ, MAP_PRIVATE, fd, i * BUFFER_SIZE);
        // But if we don't touch it, it won't be read...
        // I happen to know I have 4 KiB pages, YMMV
        for(x = 0; x < BUFFER_SIZE; x += 1024)
        {
            y += mmdata[x];
        }
        munmap(mmdata, BUFFER_SIZE);
    }
    close(fd);
#else
    int fd;
    fd = open("/dev/zero", O_RDONLY);
    for(i = 0; i < ITERATIONS; ++i)
    {
        read(fd, buffer, BUFFER_SIZE);
        for(x = 0; x < BUFFER_SIZE; x += 1024)
        {
            y += buffer[x];
        }
    }
    close(fd);

#endif

    end_time = now();
    total_time = end_time - start_time;

    printf("It took %f seconds to read 10 GiB. That's %f MiB/s.\n", total_time, ITERATIONS / total_time);

    return 0;
} 

