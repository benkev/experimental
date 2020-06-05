/* 
 * 
 * shared_mem_unlink.c: Remove the shared memory segment with van Vleck 
 * correction tables
 *
 * Build command:
 *
 * $ gcc shared_mem_unlink.c -o unlink_shm -lrt
 *
 */ 

#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>
#include <errno.h>
 
#define XY_TABLE_SHARED "/vanVleck_xy_table" 

int main(int argc, char *argv[]) {

    struct stat buf;
 

    if ( stat("/dev/shm/vanVleck_xy_table", &buf) ) {
        if (errno == ENOENT) 
            printf("Shared memory segment /dev/shm/vanVleck_xy_table " \
                   "does not exist.\n");
        else
            printf("Failure of stat(\"/dev/shm/vanVleck_xy_table\", &buf)");
        return -1;
    }

    printf("st_size = %ld\n", buf.st_size);

    if ( shm_unlink(XY_TABLE_SHARED) == -1) 
        error("shm_unlink");

    printf("Shared memory segment /dev/shm/vanVleck_xy_table " \
           "successfully removed.\n", buf.st_size);

    return 0;
}





