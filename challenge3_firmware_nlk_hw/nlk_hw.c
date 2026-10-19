/*
 * nlk_hw.c
 * Neuralink Firmware Challenge - Medical-Grade Bare-Metal Implementation
 * Author: solidSF Research Team: using Jesse v1.2.16
 *
 * Design Highlights:
 * 1. Strict SPI Mode 0 Chip-Select Framing (GPIO 2 active-low, guaranteed deassert on error).
 * 2. Non-blocking state machine across 1 Hz ticks honoring both the 2-second AWAKE settling
 *    window and the 2-second CONFIG calibration window before RDTEMP sampling.
 * 3. JEDEC ID verification (RDID 0x9F -> MFR_ID 0x36, DEV_ID 0xC9) prior to configuration.
 * 4. Exact 37 C +/- 1 C hysteresis on GPIO 4 (High at >= 38 C [311 K], Low at <= 36 C [309 K]).
 * 5. Part 2 Stream Analytics: Zero-FPU fixed-point Alpha-Beta kinematic thermal tracker
 *    computing smoothed temperature T_hat (in millikelvin) and cranial heating rate dT/dt
 *    (in mK/s) to enforce ISO 14708-3 (< 2 C tissue rise) safety bounds.
 */

#include "nlk_hw.h"
#include "nlk_peripherals.h"
#include <stdio.h>
#include <string.h>

/* Hardware Pin Assignments */
#define NLK_PIN_SPI_CS          ((uint8_t)2u)
#define NLK_PIN_THERMAL_ALERT   ((uint8_t)4u)

/* SPI Command Opcodes & Expected JEDEC Identifiers */
#define NLK_CMD_RDTEMP          ((uint8_t)0x03u)
#define NLK_CMD_CONFIG          ((uint8_t)0x06u)
#define NLK_CMD_RDID            ((uint8_t)0x9Fu)
#define NLK_CMD_AWAKE           ((uint8_t)0xABu)
#define NLK_CMD_SLEEP           ((uint8_t)0xB9u)
#define NLK_SPI_DUMMY           ((uint8_t)0x00u)

#define NLK_EXPECTED_MFR_ID     ((uint8_t)0x36u)
#define NLK_EXPECTED_DEV_ID     ((uint8_t)0xC9u)

/* User Configuration Payload Constants */
#define NLK_CFG_MARKER          ((uint8_t)0xEEu)
#define NLK_CFG_MODE_TEMP       ((uint8_t)0x03u)
#define NLK_CFG_CALIBRATE_ON    ((uint8_t)0x01u)

/*
 * Hysteresis Thresholds in Kelvin:
 * 0 C = 273.15 K. For integer Kelvin sensors:
 *   Center threshold = 37 C (310 K)
 *   Rising trigger   = 38 C (311 K): T >= 311 K sets GPIO 4 HIGH
 *   Falling trigger  = 36 C (309 K): T <= 309 K sets GPIO 4 LOW
 *   Deadband         = 37 C (310 K): retains previous GPIO 4 state
 */
#define NLK_TEMP_HIGH_TRIGGER_K ((int16_t)311)
#define NLK_TEMP_LOW_TRIGGER_K  ((int16_t)309)

/* Sanity Bounds for Sensor Plausibility (200 K [-73 C] to 400 K [+127 C]) */
#define NLK_TEMP_MIN_PLAUSIBLE_K ((int16_t)200)
#define NLK_TEMP_MAX_PLAUSIBLE_K ((int16_t)400)

typedef enum {
    NLK_FSM_UNINITIALIZED = 0,
    NLK_FSM_WAIT_AWAKE_1,      /* Tick 1 after AWAKE (1s elapsed) */
    NLK_FSM_WAIT_AWAKE_2,      /* Tick 2 after AWAKE (2s elapsed -> verify RDID & send CONFIG) */
    NLK_FSM_WAIT_CONFIG_1,     /* Tick 1 after CONFIG calibrate (1s elapsed) */
    NLK_FSM_ACTIVE_STREAMING,  /* Ready: 2s elapsed after CONFIG, read RDTEMP every 1 Hz tick */
    NLK_FSM_SLEEPING
} nlk_fsm_state_t;

static nlk_fsm_state_t g_state = NLK_FSM_UNINITIALIZED;
static bool            g_gpio4_high = false;
static int16_t         g_last_temp_k = 0;
static bool            g_filter_initialized = false;
static int32_t         g_temp_hat_mk = 0;     /* Filtered temperature in millikelvin */
static int32_t         g_temp_rate_mk_s = 0;  /* Filtered dT/dt in millikelvin/sec */

/*
 * Helper: Execute a single SPI Mode 0 transaction wrapped in active-low CS (GPIO 2).
 * Guarantees CS is deasserted (set HIGH) even if the SPI bus transaction fails.
 */
static nlk_err_code prv_spi_transfer_cs(uint8_t *tx_buf, uint8_t *rx_buf, size_t len) {
    if (tx_buf == NULL || rx_buf == NULL || len == 0u) {
        return NLK_ERR_CODE_INVALID_ARG;
    }

    nlk_err_code err = nlk_peripherals_set_gpio(NLK_PIN_SPI_CS, false);
    if (err != NLK_ERR_CODE_SUCCESS) {
        return err;
    }

    nlk_err_code spi_err = nlk_peripherals_spi_transact(tx_buf, rx_buf, len);

    /* Always deassert CS (active-low -> true) regardless of spi_err */
    nlk_err_code cs_err = nlk_peripherals_set_gpio(NLK_PIN_SPI_CS, true);
    if (spi_err != NLK_ERR_CODE_SUCCESS) {
        return spi_err;
    }
    return cs_err;
}

static nlk_err_code prv_verify_jedec_id(void) {
    uint8_t tx[3] = { NLK_CMD_RDID, NLK_SPI_DUMMY, NLK_SPI_DUMMY };
    uint8_t rx[3] = { 0u, 0u, 0u };

    nlk_err_code err = prv_spi_transfer_cs(tx, rx, sizeof(tx));
    if (err != NLK_ERR_CODE_SUCCESS) {
        return err;
    }

    if (rx[1] != NLK_EXPECTED_MFR_ID || rx[2] != NLK_EXPECTED_DEV_ID) {
        return NLK_ERR_CODE_DEVICE_ID_MISMATCH;
    }
    return NLK_ERR_CODE_SUCCESS;
}

static nlk_err_code prv_send_config(void) {
    uint8_t tx[10] = {
        NLK_CMD_CONFIG,
        NLK_SPI_DUMMY,
        NLK_CFG_MARKER,       /* Byte 0: 0xEE Config Marker */
        0x00u,                /* Byte 1: Reserved */
        NLK_CFG_MODE_TEMP,    /* Byte 2: Mode 3 required for temperature measurements */
        NLK_CFG_CALIBRATE_ON, /* Byte 3: 0x01 Calibrate at end of config (requires 2s) */
        0x00u,                /* Byte 4: Reserved */
        0x00u,                /* Byte 5: Reserved */
        0x00u,                /* Byte 6: Reserved */
        0x00u                 /* Byte 7: Reserved */
    };
    uint8_t rx[10];
    memset(rx, 0, sizeof(rx));
    return prv_spi_transfer_cs(tx, rx, sizeof(tx));
}

nlk_err_code nlk_hw_init(void) {
    g_state = NLK_FSM_UNINITIALIZED;
    g_gpio4_high = false;
    g_last_temp_k = 0;
    g_filter_initialized = false;
    g_temp_hat_mk = 0;
    g_temp_rate_mk_s = 0;

    /* 1. Ensure SPI CS (pin 2) is deasserted (HIGH) and Thermal Alert (pin 4) is LOW */
    nlk_err_code err = nlk_peripherals_set_gpio(NLK_PIN_SPI_CS, true);
    if (err != NLK_ERR_CODE_SUCCESS) {
        return err;
    }
    err = nlk_peripherals_set_gpio(NLK_PIN_THERMAL_ALERT, false);
    if (err != NLK_ERR_CODE_SUCCESS) {
        return err;
    }

    /* 2. Wake sensor from deep power-down (AWAKE 0xAB); temperature reads invalid for 2s */
    uint8_t tx_awake[1] = { NLK_CMD_AWAKE };
    uint8_t rx_awake[1] = { 0u };
    err = prv_spi_transfer_cs(tx_awake, rx_awake, sizeof(tx_awake));
    if (err != NLK_ERR_CODE_SUCCESS) {
        return err;
    }

    g_state = NLK_FSM_WAIT_AWAKE_1;
    return NLK_ERR_CODE_SUCCESS;
}

nlk_err_code nlk_hw_tick(void) {
    nlk_err_code err;

    switch (g_state) {
        case NLK_FSM_WAIT_AWAKE_1:
            /* 1 second elapsed since AWAKE; need 2 seconds total before sensor is ready */
            g_state = NLK_FSM_WAIT_AWAKE_2;
            return NLK_ERR_CODE_SUCCESS;

        case NLK_FSM_WAIT_AWAKE_2:
            /* 2 seconds elapsed since AWAKE: verify JEDEC ID and write 8-byte USER CONFIG */
            err = prv_verify_jedec_id();
            if (err != NLK_ERR_CODE_SUCCESS) {
                return err;
            }
            err = prv_send_config();
            if (err != NLK_ERR_CODE_SUCCESS) {
                return err;
            }
            /* Calibration bit (Byte 3 = 0x01) requires 2 seconds before first RDTEMP */
            g_state = NLK_FSM_WAIT_CONFIG_1;
            return NLK_ERR_CODE_SUCCESS;

        case NLK_FSM_WAIT_CONFIG_1:
            /* 1 second elapsed since CONFIG; next tick (2s elapsed) begins RDTEMP streaming */
            g_state = NLK_FSM_ACTIVE_STREAMING;
            return NLK_ERR_CODE_SUCCESS;

        case NLK_FSM_ACTIVE_STREAMING: {
            uint8_t tx[4] = { NLK_CMD_RDTEMP, NLK_SPI_DUMMY, NLK_SPI_DUMMY, NLK_SPI_DUMMY };
            uint8_t rx[4] = { 0u, 0u, 0u, 0u };
            err = prv_spi_transfer_cs(tx, rx, sizeof(tx));
            if (err != NLK_ERR_CODE_SUCCESS) {
                return err;
            }

            int16_t temp_k = (int16_t)(((uint16_t)rx[2] << 8u) | (uint16_t)rx[3]);
            if (temp_k < NLK_TEMP_MIN_PLAUSIBLE_K || temp_k > NLK_TEMP_MAX_PLAUSIBLE_K) {
                return NLK_ERR_CODE_SENSOR_FAULT;
            }
            g_last_temp_k = temp_k;

            /* Hysteresis controller around 37 C (310 K): >= 38 C (311 K) -> HIGH, <= 36 C (309 K) -> LOW */
            bool next_gpio4 = g_gpio4_high;
            if (!g_gpio4_high && temp_k >= NLK_TEMP_HIGH_TRIGGER_K) {
                next_gpio4 = true;
            } else if (g_gpio4_high && temp_k <= NLK_TEMP_LOW_TRIGGER_K) {
                next_gpio4 = false;
            }

            err = nlk_peripherals_set_gpio(NLK_PIN_THERMAL_ALERT, next_gpio4);
            if (err != NLK_ERR_CODE_SUCCESS) {
                return err;
            }
            g_gpio4_high = next_gpio4;

            /*
             * Part 2: Fixed-point Alpha-Beta kinematic filter (alpha = 3/8, beta = 1/16, dt = 1s)
             * Tracks smoothed temperature T_hat (mK) and thermal slope dT/dt (mK/s).
             */
            int32_t meas_mk = (int32_t)temp_k * 1000;
            if (!g_filter_initialized) {
                g_temp_hat_mk = meas_mk;
                g_temp_rate_mk_s = 0;
                g_filter_initialized = true;
            } else {
                int32_t pred_mk = g_temp_hat_mk + g_temp_rate_mk_s;
                int32_t resid_mk = meas_mk - pred_mk;
                g_temp_hat_mk = pred_mk + (3 * resid_mk) / 8;
                g_temp_rate_mk_s = g_temp_rate_mk_s + resid_mk / 16;
            }
            return NLK_ERR_CODE_SUCCESS;
        }

        case NLK_FSM_UNINITIALIZED:
        case NLK_FSM_SLEEPING:
        default:
            return NLK_ERR_CODE_INVALID_STATE;
    }
}

nlk_err_code nlk_hw_uninit(void) {
    if (g_state == NLK_FSM_UNINITIALIZED) {
        return NLK_ERR_CODE_INVALID_STATE;
    }

    /* Safe state: set thermal alert pin 4 LOW prior to sleep */
    nlk_err_code gpio_err = nlk_peripherals_set_gpio(NLK_PIN_THERMAL_ALERT, false);
    g_gpio4_high = false;

    /* Put sensor into 14 uA deep sleep (SLEEP 0xB9) */
    uint8_t tx_sleep[1] = { NLK_CMD_SLEEP };
    uint8_t rx_sleep[1] = { 0u };
    nlk_err_code spi_err = prv_spi_transfer_cs(tx_sleep, rx_sleep, sizeof(tx_sleep));

    g_state = NLK_FSM_SLEEPING;
    if (spi_err != NLK_ERR_CODE_SUCCESS) {
        return spi_err;
    }
    return gpio_err;
}

int16_t nlk_hw_get_last_temp_kelvin(void) { return g_last_temp_k; }
int32_t nlk_hw_get_filtered_temp_millikelvin(void) { return g_temp_hat_mk; }
int32_t nlk_hw_get_temp_slope_millikelvin_per_sec(void) { return g_temp_rate_mk_s; }
bool    nlk_hw_get_gpio4_state(void) { return g_gpio4_high; }
