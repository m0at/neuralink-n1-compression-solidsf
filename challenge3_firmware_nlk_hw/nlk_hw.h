/*
 * nlk_hw.h
 * Neuralink Firmware Challenge - Hardware Entry Points
 */
#ifndef NLK_HW_H
#define NLK_HW_H

#include <stdbool.h>
#include <stdint.h>
#include "nlk_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Required entry points per Neuralink Firmware Challenge specification */
nlk_err_code nlk_hw_init(void);
nlk_err_code nlk_hw_tick(void);
nlk_err_code nlk_hw_uninit(void);

/* Part 2 telemetry & diagnostic getters for unit verification */
int16_t nlk_hw_get_last_temp_kelvin(void);
int32_t nlk_hw_get_filtered_temp_millikelvin(void);
int32_t nlk_hw_get_temp_slope_millikelvin_per_sec(void);
bool    nlk_hw_get_gpio4_state(void);

#ifdef __cplusplus
}
#endif

#endif /* NLK_HW_H */
