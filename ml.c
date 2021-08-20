#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <openssl/md5.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>

// struct Node;
typedef struct Node
{
    uint64_t weight;
    struct Node *next;
    uint8_t *word;
} Node;

// int parse_args (int argc, char *argv[])
// {
//     int flags, opt;
//     int nsecs, tfnd;

//     nsecs = 0;
//     tfnd = 0;
//     flags = 0;
//     while ((opt = getopt(argc, argv, "nt:")) != -1) 
//     {
//         switch (opt) 
//         {
//         case 'n':
//             flags = 1;
//             break;
//         case 't':
//             nsecs = atoi(optarg);
//             tfnd = 1;
//             break;
//         default: /* '?' */
//             fprintf(stderr, "Usage: %s [-t nsecs] [-n] name\n", argv[0]);
//             exit(EXIT_FAILURE);
//         }
//     }

//     printf("flags=%d; tfnd=%d; nsecs=%d; optind=%d\n",
//             flags, tfnd, nsecs, optind);

//     if (optind >= argc) {
//         fprintf(stderr, "Expected argument after options\n");
//         exit(EXIT_FAILURE);
//     }

//     printf("name argument = %s\n", argv[optind]);

//     /* Other code omitted */

//     exit(EXIT_SUCCESS);
// }

uint64_t nextPrime(uint64_t n)
{
    uint64_t i;
    uint64_t root;
    uint8_t keepGoing = 1;

    if (n % 2 == 0)
    {
        n++;
    }

    while (keepGoing)
    {
        keepGoing = 0;
        root = sqrt(n);

        for (i = 3; i <= root; i++)
        {
            if (n % i == 0)
            {
                // Move on to the next candidate for primality. Since n is odd, we
                // don't want to increment it by 1. That would give us an even
                // integer greater than 2, which would necessarily be non-prime.
                n += 2;
                keepGoing = 1;

                // Leave for-loop. Move on to next iteration of while-loop.
                break;
            }
        }
    }

    return n;   
}

void merge(Node** arr, int l, int m, int r) 
{ 
    int i, j, k; 
    int n1 = m - l + 1; 
    int n2 = r - m; 
  
    /* create temp arrays */
    Node *L[n1], *R[n2]; 
  
    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++) 
        L[i] = arr[l + i]; 
    for (j = 0; j < n2; j++) 
        R[j] = arr[m + 1 + j]; 
  
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray 
    j = 0; // Initial index of second subarray 
    k = l; // Initial index of merged subarray 
    while (i < n1 && j < n2) { 
        if (L[i]->weight <= R[j]->weight) { 
            arr[k] = L[i]; 
            i++; 
        } 
        else { 
            arr[k] = R[j]; 
            j++; 
        } 
        k++; 
    } 
  
    /* Copy the remaining elements of L[], if there 
       are any */
    while (i < n1) { 
        arr[k] = L[i]; 
        i++; 
        k++; 
    } 
  
    /* Copy the remaining elements of R[], if there 
       are any */
    while (j < n2) { 
        arr[k] = R[j]; 
        j++; 
        k++; 
    } 
} 
  
/* l is for left index and r is right index of the 
   sub-array of arr to be sorted */
void mergeSort(Node** arr, int l, int r) 
{ 
    if (l < r) 
    { 
        // Same as (l+r)/2, but avoids overflow for 
        // large l and h 
        int m = l + (r - l) / 2; 
// printf("%d\n", l);
// printf("%d\n\n", r);
        // Sort first and second halves 
        mergeSort(arr, l, m); 
        mergeSort(arr, m + 1, r); 
 
        merge(arr, l, m, r); 
    } 
} 

uint8_t hash (uint64_t word_count, Node* node)
{
    uint8_t retval = EXIT_FAILURE;
    MD5_CTX ctx;
    uint8_t out[MD5_DIGEST_LENGTH];
    uint64_t table_size = nextPrime(word_count * 2);
    // char** argv2 = calloc(table_size, sizeof(char*));
    uint64_t colls = 0;
    Node **hm = NULL;
    uint64_t i;
    uint64_t k;

    if (NULL == (hm = malloc(table_size * sizeof(Node*))))
    {
        printf("malloc failed\n");
        goto malloc_error;
    }
    memset(hm, 0, table_size * sizeof(Node*));

    for (i = 0; NULL != node; node = node->next, i++)
    {
        MD5_Init(&ctx);
        MD5_Update(&ctx, node->word, strlen(node->word));
        MD5_Final(out, &ctx);

        for (k = 0; NULL != hm[(*((uint16_t*)out) + (uint64_t)pow(2, k)) % table_size]; k++)
        {
            if (0 == strcmp(hm[(*((uint16_t*)out) + (uint64_t)pow(2, k)) % table_size]->word, node->word))
            {
                hm[(*((uint16_t*)out) + (uint64_t)pow(2, k)) % table_size]->weight++;
                break;
            }
            colls++;
        }

        if (NULL == hm[(*((uint16_t*)out) + (uint64_t)pow(2, k)) % table_size])
        {
            node->weight++;
            hm[(*((uint16_t*)out) + (uint64_t)pow(2, k)) % table_size] = node;
        }
    }

    k = table_size - 1;
    for (i = 0; i < table_size; i++)
    {
        if (NULL == hm[i])
        {
            for (k = k; k > i; k--)
            {
                if (NULL != hm[k])
                {
                    hm[i] = hm[k];
                    hm[k] = NULL;
                    break;
                }
            }
        }
    }

    for (i = 0; NULL != hm[i]; i++);

    mergeSort(hm, 0, i - 1);

    for (i = 0; i < table_size; i++)
    {
        if (NULL != hm[i])
        {
            printf("%lu occurrences of %s\n", hm[i]->weight,hm[i]->word);
        }
    }

    printf("Total Collisions: %lu\n", colls);
    printf("Total Items: %lu\n", word_count);
    printf("Total Space: %lu\n", table_size);

    goto end_success;

malloc_error:
end_success:
    return retval;
}


int main(int argc, char const *argv[])
{
    struct stat statbuf;
    int32_t fd = -1;
    uint8_t retval = EXIT_FAILURE;
    uint8_t *str = NULL;
    uint64_t i;
    uint64_t word_count = 0;
    uint64_t k;
    Node *tail = NULL;
    Node *node = NULL;
    Node *node_last = NULL;

    if (-1 == (fd = open(argv[1], O_RDONLY)))
    {
        printf("open failed\n");
        goto open_error;
    }

    memset(&statbuf, 0, sizeof(stat));
    if (0 != fstat(fd, &statbuf))
    {
        printf("fstat failed\n");
        goto fstat_error;
    }

    if (NULL == (str = malloc(sizeof(uint8_t)*(statbuf.st_size + 1))))
    {
        printf("malloc failed\n");
        goto malloc_error;
    }

    memset(str, 0, sizeof(uint8_t)*(statbuf.st_size + 1));
    if (-1 == read(fd, str, statbuf.st_size))
    {
        printf("read error\n");
        goto read_error;
    }

    for (i = 0; i < statbuf.st_size; i++)
    {
        if (0 != isalpha(str[i]))
        {
            word_count++;
            if (NULL == (node = malloc(sizeof(Node))))
            {
                printf("malloc failed 3\n");
                goto malloc_error_3;
            }
            memset(node, 0, sizeof(Node));

            if (NULL == node_last)
            {
                tail = node;
            }
            else
            {
                node_last->next = node;
            }

            for (k = i; ('\0' != str[k]) && isalpha(str[k]); k++);

            if (NULL == (node->word = malloc(sizeof(uint8_t)*(k - i + 1))))
            {
                printf("malloc failed 3\n");
                goto malloc_error_3;
            }
            memset(node->word, 0, k - i + 1);

            memcpy(node->word, str + i, k - i);

            node_last = node;
            i += k - i;
        }
    }
printf("before hash 3\n");
    hash(word_count, tail);

    goto end_success;

malloc_error_3:
    free(tail);
malloc_error_2:
read_error:
    free(str);
malloc_error:
fstat_error:
open_error:
end_success:
    return retval;
}