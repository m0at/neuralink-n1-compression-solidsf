/*
 * nlk_errors.h
 * Neuralink Firmware Challenge - Error Code Definitions
 */
#ifndef NLK_ERRORS_H
#define NLK_ERRORS_H

#include <stdint.h>

typedef int nlk_err_code;

#define NLK_ERR_CODE_SUCCESS            ((nlk_err_code)0)
#define NLK_ERR_CODE_UNKNOWN            ((nlk_err_code)1)
#define NLK_ERR_CODE_TIMEOUT            ((nlk_err_code)2)
#define NLK_ERR_CODE_INVALID_ARG        ((nlk_err_code)3)
#define NLK_ERR_CODE_DEVICE_ID_MISMATCH ((nlk_err_code)4)
#define NLK_ERR_CODE_INVALID_STATE      ((nlk_err_code)5)
#define NLK_ERR_CODE_SENSOR_FAULT       ((nlk_err_code)6)

#endif /* NLK_ERRORS_H */
