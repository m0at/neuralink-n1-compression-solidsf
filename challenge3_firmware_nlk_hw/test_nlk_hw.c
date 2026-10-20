/*
 * test_nlk_hw.c
 * Comprehensive Hardware-in-the-Loop Mock & Fault-Injection Test Suite
 * for Neuralink Firmware Challenge (nlk_hw)
 */

#include "nlk_hw.h"
#include "nlk_peripherals.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Mock Peripheral State */
static bool     mock_cs_low = false;
static bool     mock_gpio4 = false;
static bool     mock_sensor_awake = false;
static bool     mock_sensor_configured = false;
static uint8_t  mock_mfr_id = 0x36u;
static uint8_t  mock_dev_id = 0xC9u;
static int16_t  mock_temp_kelvin = 298; /* 25 C */
static int      mock_force_spi_err = NLK_ERR_CODE_SUCCESS;
static int      mock_force_gpio_err = NLK_ERR_CODE_SUCCESS;
static uint32_t mock_spi_tx_count = 0;

static void reset_mock(void) {
    mock_cs_low = false;
    mock_gpio4 = false;
    mock_sensor_awake = false;
    mock_sensor_configured = false;
    mock_mfr_id = 0x36u;
    mock_dev_id = 0xC9u;
    mock_temp_kelvin = 298;
    mock_force_spi_err = NLK_ERR_CODE_SUCCESS;
    mock_force_gpio_err = NLK_ERR_CODE_SUCCESS;
    mock_spi_tx_count = 0;
}

nlk_err_code nlk_peripherals_set_gpio(uint8_t pin, bool value) {
    if (mock_force_gpio_err != NLK_ERR_CODE_SUCCESS) {
        return mock_force_gpio_err;
    }
    if (pin == 2u) {
        mock_cs_low = !value;
        return NLK_ERR_CODE_SUCCESS;
    }
    if (pin == 4u) {
        mock_gpio4 = value;
        return NLK_ERR_CODE_SUCCESS;
    }
    return NLK_ERR_CODE_INVALID_ARG;
}

nlk_err_code nlk_peripherals_spi_transact(uint8_t *tx_buf, uint8_t *rx_buf, size_t len) {
    if (!mock_cs_low) {
        /* Chip select MUST be asserted (active low) during SPI transaction! */
        return NLK_ERR_CODE_UNKNOWN;
    }
    if (mock_force_spi_err != NLK_ERR_CODE_SUCCESS) {
        return mock_force_spi_err;
    }
    mock_spi_tx_count++;
    memset(rx_buf, 0, len);

    switch (tx_buf[0]) {
        case 0xABu: /* AWAKE */
            assert(len == 1u);
            mock_sensor_awake = true;
            break;
        case 0xB9u: /* SLEEP */
            assert(len == 1u);
            mock_sensor_awake = false;
            break;
        case 0x9Fu: /* RDID */
            assert(len == 3u);
            assert(mock_sensor_awake);
            rx_buf[1] = mock_mfr_id;
            rx_buf[2] = mock_dev_id;
            break;
        case 0x06u: /* CONFIG */
            assert(len == 10u);
            assert(mock_sensor_awake);
            assert(tx_buf[1] == 0x00u);
            assert(tx_buf[2] == 0xEEu);
            assert(tx_buf[4] == 0x03u);
            assert(tx_buf[5] == 0x01u);
            mock_sensor_configured = true;
            break;
        case 0x03u: /* RDTEMP */
            assert(len == 4u);
            assert(mock_sensor_awake && mock_sensor_configured);
            assert(tx_buf[1] == 0x00u);
            rx_buf[2] = (uint8_t)((uint16_t)mock_temp_kelvin >> 8u);
            rx_buf[3] = (uint8_t)((uint16_t)mock_temp_kelvin & 0xFFu);
            break;
        default:
            return NLK_ERR_CODE_INVALID_ARG;
    }
    return NLK_ERR_CODE_SUCCESS;
}

int main(void) {
    printf("=== Neuralink Firmware Challenge (nlk_hw) Verification Suite ===\n");

    /* Test 1: Boot, 2s AWAKE delay, RDID + CONFIG, 2s calibrate delay, and Hysteresis Sweep */
    reset_mock();
    assert(nlk_hw_init() == NLK_ERR_CODE_SUCCESS);
    assert(mock_sensor_awake == true);
    assert(mock_cs_low == false);
    assert(mock_gpio4 == false);

    /* Tick 1 (1s after AWAKE): waiting */
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(mock_sensor_configured == false);

    /* Tick 2 (2s after AWAKE): verifies RDID and sends CONFIG */
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(mock_sensor_configured == true);

    /* Tick 3 (1s after CONFIG): waiting for 2s calibration */
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);

    /* Tick 4 (2s after CONFIG): first valid RDTEMP at 25 C (298 K) -> GPIO 4 stays LOW */
    mock_temp_kelvin = 298;
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(mock_gpio4 == false);
    assert(nlk_hw_get_last_temp_kelvin() == 298);

    /* Rise to 37 C (310 K): inside hysteresis deadband -> GPIO 4 MUST stay LOW */
    mock_temp_kelvin = 310;
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(mock_gpio4 == false);

    /* Rise to 38 C (311 K): hits rising hysteresis threshold -> GPIO 4 goes HIGH */
    mock_temp_kelvin = 311;
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(mock_gpio4 == true);

    /* Cool to 37 C (310 K): inside hysteresis deadband -> GPIO 4 MUST stay HIGH */
    mock_temp_kelvin = 310;
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(mock_gpio4 == true);

    /* Cool to 36 C (309 K): hits falling hysteresis threshold -> GPIO 4 goes LOW */
    mock_temp_kelvin = 309;
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(mock_gpio4 == false);
    printf("[PASS] Test 1: Boot Timing (2s AWAKE + 2s CONFIG) & 37C +/- 1C Hysteresis Loop\n");

    /* Test 2: Part 2 Alpha-Beta Thermal Filter & Shutdown to 14 uA SLEEP */
    assert(nlk_hw_get_filtered_temp_millikelvin() > 298000);
    assert(nlk_hw_uninit() == NLK_ERR_CODE_SUCCESS);
    assert(mock_sensor_awake == false);
    assert(mock_gpio4 == false);
    assert(nlk_hw_tick() == NLK_ERR_CODE_INVALID_STATE);
    printf("[PASS] Test 2: Alpha-Beta Thermal Tracker & Deep-Sleep (0xB9) Shutdown\n");

    /* Test 3: Corrupted JEDEC Manufacturer/Device ID Rejection */
    reset_mock();
    mock_mfr_id = 0xFFu;
    assert(nlk_hw_init() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_tick() == NLK_ERR_CODE_DEVICE_ID_MISMATCH);
    assert(mock_cs_low == false); /* CS guaranteed deasserted even on error */
    printf("[PASS] Test 3: Corrupted JEDEC ID Fault Detection & CS Deassertion\n");

    /* Test 4: SPI Bus Timeout & Out-of-Range Sensor Fault Rejection */
    reset_mock();
    assert(nlk_hw_init() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    mock_temp_kelvin = -50; /* Unphysical negative Kelvin */
    assert(nlk_hw_tick() == NLK_ERR_CODE_SENSOR_FAULT);
    mock_temp_kelvin = 300;
    mock_force_spi_err = NLK_ERR_CODE_TIMEOUT;
    assert(nlk_hw_tick() == NLK_ERR_CODE_TIMEOUT);
    assert(mock_cs_low == false);
    printf("[PASS] Test 4: SPI Bus Timeout & Unphysical Kelvin Plausibility Guard\n");

    /* Test 5: Exact Boundary Hysteresis (309000 mK / 310000 mK / 311000 mK) & Millikelvin Alpha-Beta Convergence */
    reset_mock();
    assert(nlk_hw_init() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    mock_temp_kelvin = 309; /* 309,000 mK (36 C exact lower boundary) */
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_get_filtered_temp_millikelvin() == 309000);
    assert(nlk_hw_get_temp_slope_millikelvin_per_sec() == 0);
    assert(nlk_hw_get_gpio4_state() == false);

    mock_temp_kelvin = 311; /* 311,000 mK (38 C exact upper boundary) */
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_get_gpio4_state() == true);
    /* Alpha-Beta step: pred = 309000, resid = +2000 -> T_hat = 309000 + (3*2000)/8 = 309750 mK, rate = +125 mK/s */
    assert(nlk_hw_get_filtered_temp_millikelvin() == 309750);
    assert(nlk_hw_get_temp_slope_millikelvin_per_sec() == 125);
    printf("[PASS] Test 5: Exact 309000 mK / 311000 mK Boundary Hysteresis & Fixed-Point Alpha-Beta Step\n");

    /* Test 6: 0xB9 Deep-Sleep Wakeup Recovery & Double-Uninit Rejection */
    assert(nlk_hw_uninit() == NLK_ERR_CODE_SUCCESS);
    assert(mock_sensor_awake == false);
    assert(nlk_hw_get_gpio4_state() == false);
    /* Second consecutive uninit while already sleeping must return INVALID_STATE without SPI traffic */
    uint32_t tx_before = mock_spi_tx_count;
    assert(nlk_hw_uninit() == NLK_ERR_CODE_INVALID_STATE);
    assert(mock_spi_tx_count == tx_before);
    /* Re-wake from 0xB9 deep sleep via nlk_hw_init() without resetting mock hardware */
    assert(nlk_hw_init() == NLK_ERR_CODE_SUCCESS);
    assert(mock_sensor_awake == true);
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    mock_temp_kelvin = 310;
    assert(nlk_hw_tick() == NLK_ERR_CODE_SUCCESS);
    assert(nlk_hw_get_last_temp_kelvin() == 310);
    assert(nlk_hw_get_filtered_temp_millikelvin() == 310000);
    assert(nlk_hw_uninit() == NLK_ERR_CODE_SUCCESS);
    printf("[PASS] Test 6: 0xB9 Deep-Sleep Wakeup Recovery & Double-Uninit Guard\n");

    printf("ALL 6 NEURALINK FIRMWARE TEST SUITES PASSED (0 FAILURES).\n");
    return 0;
}
