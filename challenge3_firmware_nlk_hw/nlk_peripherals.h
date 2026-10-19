/*
 * nlk_peripherals.h
 * Neuralink Firmware Challenge - Peripheral Interface
 */
#ifndef NLK_PERIPHERALS_H
#define NLK_PERIPHERALS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "nlk_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

nlk_err_code nlk_peripherals_set_gpio(uint8_t pin, bool value);
nlk_err_code nlk_peripherals_spi_transact(uint8_t *tx_buf, uint8_t *rx_buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* NLK_PERIPHERALS_H */
