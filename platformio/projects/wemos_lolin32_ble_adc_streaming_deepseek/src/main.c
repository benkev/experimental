#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_adc/adc_oneshot.h"  // Modern ADC driver
#include "esp_adc/adc_cali.h"     // Modern calibration
#include "esp_adc/adc_cali_scheme.h"
#include "esp_nimble_hci.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"

#define TAG "BLE_ADC"

/* EXACT UUIDs matching your Arduino version */
static const ble_uuid128_t gatt_svc_uuid = 
    BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
                     0x93, 0xf3, 0xa3, 0xb5, 0x01, 0x00, 0x40, 0x6e);
                     
static const ble_uuid128_t gatt_chr_uuid = 
    BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
                     0x93, 0xf3, 0xa3, 0xb5, 0x03, 0x00, 0x40, 0x6e);

/* ADC Configuration */
#define ADC_UNIT          ADC_UNIT_1
#define ADC_CHANNEL       ADC_CHANNEL_0  // GPIO36
#define ADC_ATTEN         ADC_ATTEN_DB_11
#define ADC_BITWIDTH      ADC_BITWIDTH_12

static uint16_t conn_handle;
static bool subscribed = false;
static uint16_t chr_val_handle;
static adc_oneshot_unit_handle_t adc_handle = NULL;
static adc_cali_handle_t cali_handle = NULL;

/* Initialize modern ADC with calibration */
void adc_init() {
    // 1. ADC unit config
    adc_oneshot_unit_init_cfg_t unit_config = {
        .unit_id = ADC_UNIT,
        .ulp_mode = ADC_ULP_MODE_DISABLE,
    };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_config, &adc_handle));

    // 2. ADC channel config
    adc_oneshot_chan_cfg_t channel_config = {
        .atten = ADC_ATTEN,
        .bitwidth = ADC_BITWIDTH,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(adc_handle, ADC_CHANNEL, &channel_config));

    // 3. Calibration (optional but recommended)
    adc_cali_line_fitting_config_t cali_config = {
        .unit_id = ADC_UNIT,
        .atten = ADC_ATTEN,
        .bitwidth = ADC_BITWIDTH,
    };
    if (adc_cali_create_scheme_line_fitting(&cali_config, &cali_handle) != ESP_OK) {
        ESP_LOGW(TAG, "No calibration scheme available, using raw values");
        cali_handle = NULL;
    }
}

/* Clean up ADC resources */
void adc_deinit() {
    if (cali_handle) {
        adc_cali_delete_scheme_line_fitting(cali_handle);
    }
    adc_oneshot_del_unit(adc_handle);
}

/* Read ADC with optional calibration */
uint16_t read_adc() {
    int raw;
    ESP_ERROR_CHECK(adc_oneshot_read(adc_handle, ADC_CHANNEL, &raw));
    
    if (cali_handle) {
        int voltage;
        ESP_ERROR_CHECK(adc_cali_raw_to_voltage(cali_handle, raw, &voltage));
        return (uint16_t)voltage;
    }
    return (uint16_t)raw;
}

/* [Rest of the BLE code remains exactly the same as previous example...] */

void app_main() {
    nvs_flash_init();
    adc_init();  // Modern ADC initialization
    
    // BLE initialization
    esp_nimble_hci_and_controller_init();
    nimble_port_init();
    ble_hs_cfg.sync_cb = ble_on_sync;
    nimble_port_freertos_init(ble_host_task);
    
    while(1) {
        if(subscribed) {
            uint16_t adc = read_adc();  // Using modern ADC read
            ble_gattc_notify(conn_handle, chr_val_handle, sizeof(adc), &adc);
            vTaskDelay(pdMS_TO_TICKS(10));
        } else {
            vTaskDelay(pdMS_TO_TICKS(500));
        }
    }
    
    adc_deinit();  // Cleanup on exit (though we never get here)
}