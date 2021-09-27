#include <ctype.h>
#include <dirent.h> 
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <openssl/md5.h>
#include <sodium.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>

#define MESSAGE ((const unsigned char *) "test")
#define MESSAGE_LEN 4
#define CIPHERTEXT_LEN (crypto_secretbox_MACBYTES + MESSAGE_LEN)

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

uint8_t check_file (const char *path, uint8_t *is_regular_file)
{
    uint8_t retval = EXIT_FAILURE;
    struct stat path_stat;

    if (0 != stat(path, &path_stat))
    {
        printf("stat failed\n");
        goto stat_error;
    }

    *is_regular_file = S_ISREG(path_stat.st_mode);

    retval = EXIT_SUCCESS;
    goto end_success;

end_success:
stat_error:
    return retval;
}

// int check_file (const uint8_t *path)
// {
//     struct stat path_stat;

//     if (0 != stat(path, &path_stat))
//     {
//         printf("stat failed\n");
//     }

//     return S_ISREG(path_stat.st_mode);
// }

uint64_t next_prime(uint64_t n)
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
    uint64_t table_size = next_prime(word_count * 2);
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

uint8_t reader (uint8_t** buf, const uint8_t *path)
{
    struct stat statbuf;
    int32_t fd = -1;
    uint8_t retval = EXIT_FAILURE;

    if (-1 == (fd = open(path, O_RDONLY)))
    {
        printf("%d: open failed\n", __LINE__);
        goto open_error;
    }

    memset(&statbuf, 0, sizeof(stat));
    if (0 != fstat(fd, &statbuf))
    {
        printf("fstat failed\n");
        goto fstat_error;
    }

    *buf = malloc(sizeof(uint8_t)*(statbuf.st_size + 1));
    if (NULL == *buf)
    {
        printf("malloc failed\n");
        goto malloc_error;
    }

    memset(*buf, 0, sizeof(uint8_t)*(statbuf.st_size + 1));
    errno = 0;
    if (-1 == read(fd, *buf, statbuf.st_size))
    {
        printf("read failed : %s\n", strerror(errno));
        goto read_error;
    }

    close(fd);

    retval = EXIT_SUCCESS;
    goto end_success;

read_error:
    free(*buf);
malloc_error:
fstat_error:
    close(fd);
open_error:
end_success:
    return retval;
}

// TODO: make reliable
uint8_t writer (uint8_t* buf, uint64_t buf_len, const uint8_t *path)
{
    int32_t fd = -1;
    uint8_t retval = EXIT_FAILURE;

    errno = 0;
    if (-1 == (fd = open(path, O_CREAT | O_WRONLY)))
    {
        printf("%d: open failed - %s\n", __LINE__, strerror(errno));
        goto open_error;
    }

    if (-1 == write(fd, buf, buf_len))
    {
        printf("write failed\n");
        goto write_error;
    }

    close(fd);

    retval = EXIT_SUCCESS;
    goto end_success;

write_error:
    close(fd);
open_error:
end_success:
    return retval;
}

uint8_t encrypt_file (uint8_t *path, uint8_t *key, uint8_t *encrypted_path)
{
    uint8_t retval = EXIT_FAILURE;
    uint8_t *buf = NULL;
    uint64_t buf_len;
    uint8_t nonce[crypto_secretbox_NONCEBYTES];
    uint8_t *ciphertext;
    uint64_t ciphertext_len;

    if (NULL == path)
    {
        printf("path == NULL\n");
        goto path_error;
    }

    if (key == NULL)
    {
        printf("key == NULL\n");
        goto key_null_error;
    }

    if (crypto_secretbox_KEYBYTES != strnlen(key, crypto_secretbox_KEYBYTES))
    {
        printf("key != crypto_secretbox_KEYBYTES. Key is: '%s'\n", key);
        goto key_len_error;
    }

    if (EXIT_SUCCESS != reader(&buf, path))
    {
        printf("reader failed\n");
        goto reader_error;
    }

    buf_len = strlen(buf);
    ciphertext_len = crypto_secretbox_MACBYTES + buf_len;

    if (-1 == sodium_init()) 
    {
        printf("sodium_init failed\n");
        goto sodium_init_error;
    }

    ciphertext = malloc(ciphertext_len);
    if (NULL == ciphertext)
    {
        printf("malloc failed\n");
        goto malloc_error;
    }

    crypto_secretbox_keygen(key);
    randombytes_buf(nonce, sizeof nonce);
    crypto_secretbox_easy(ciphertext, buf, buf_len, nonce, key);

    if (NULL == encrypted_path)
    {
        encrypted_path = malloc(strlen("ENCRYPTED_") + strnlen(path, FILENAME_MAX));
        if (NULL == encrypted_path)
        {
            printf("malloc failed 2\n");
            goto malloc_error_2;
        }
        sprintf(encrypted_path, "ENCRYPTED_");
        strncat(encrypted_path, path, FILENAME_MAX);
    }

    if (EXIT_SUCCESS != writer(ciphertext, ciphertext_len, encrypted_path))
    {
        printf("writer failed\n");
        goto writer_error;
    }
    
    //TODO: implement decryption
    // unsigned char decrypted[MESSAGE_LEN];
    // if (0 != crypto_secretbox_open_easy(decrypted, ciphertext, CIPHERTEXT_LEN, nonce, key)) 
    // {
    //     /* message forged! */
    // }

    retval = EXIT_SUCCESS;
    goto end_success;

writer_error:
crypto_secretbox_open_easy_error:
    free(ciphertext);
malloc_error_2:
malloc_error:
sodium_init_error:
    free(buf);
reader_error:
key_len_error:
key_null_error:
path_error:
end_success:
    return retval;
}

uint8_t encrypt_dir (const uint8_t *path, uint8_t *key) 
{
    uint8_t retval = EXIT_FAILURE;
    uint8_t is_regular_file = 0;
    DIR *d;
    struct dirent *dir;

    d = opendir(path);
    if (NULL == d) 
    {
        printf("opendir failed\n");
        goto opendir_error;
    }

    errno = 0;
    while ((dir = readdir(d)) != NULL) 
    {
        if (0 != errno)
        {
            printf("readdir failed\n");
            goto readdir_error;
        }

        if (EXIT_SUCCESS != check_file(dir->d_name, &is_regular_file))
        {
            printf("readdir failed\n");
            goto readdir_error;
        }

        if (1 == is_regular_file)
        {
            if (EXIT_SUCCESS != encrypt_file(dir->d_name, key, NULL))
            {
                printf("encrypt_file failed while encrypting '%s'\n", dir->d_name);
                continue;
            }
            printf("Encrypted file : '%s'\n", dir->d_name);
        }
        else
        {
            printf("'%s' is not a regular file\n", dir->d_name);
        }
    }
    closedir(d);

    retval = EXIT_SUCCESS;
    goto end_success;

encrypt_file_error:
readdir_error:
    closedir(d);
opendir_error:
end_success:
    return retval;
}

int main (int argc, char const *argv[])
{
    uint8_t retval = EXIT_FAILURE;
    uint8_t *buf = NULL;
    uint64_t i;
    uint64_t word_count = 0;
    uint64_t k;
    uint64_t buf_len;
    Node *tail = NULL;
    Node *node = NULL;
    Node *node_last = NULL;
    uint8_t key[crypto_secretbox_KEYBYTES] = "12345678123456781234567812345678";

    if (EXIT_SUCCESS != reader(&buf, argv[1]))
    {
        printf("reader failed\n");
        goto reader_error;
    }

    buf_len = strlen(buf);

    for (i = 0; i < buf_len; i++)
    {
        if (0 != isalpha(buf[i]))
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

            for (k = i; ('\0' != buf[k]) && isalpha(buf[k]); k++);

            if (NULL == (node->word = malloc(sizeof(uint8_t)*(k - i + 1))))
            {
                printf("malloc failed 3\n");
                goto malloc_error_3;
            }
            memset(node->word, 0, k - i + 1);

            memcpy(node->word, buf + i, k - i);

            node_last = node;
            i += k - i;
        }
    }
printf("before hash 3\n");
    hash(word_count, tail);
    
    printf("%s\n", key);

    if (EXIT_SUCCESS != encrypt_dir(argv[2], key))
    {
        printf("encrypt_dir failed\n");
        goto encrypt_dir_error;
    }

    retval = EXIT_SUCCESS;
    goto end_success;

encrypt_dir_error:
malloc_error_3:
    free(tail);
malloc_error_2:
reader_error:
    free(buf);
end_success:
    return retval;
}