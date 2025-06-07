#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

static const char *TAG = "BLE_ADC";

static uint16_t conn_handle = BLE_HS_CONN_HANDLE_NONE;
static bool subscribed = false;
static uint16_t adc_value = 0;

#define SERVICE_UUID        0xFFF0
#define CHARACTERISTIC_UUID 0xFFF1

static const ble_uuid128_t service_uuid =
    BLE_UUID128_INIT(0x6e, 0x40, 0x00, 0x01, 0xb5, 0xa3, 0xf3, 0x93,
                     0xe0, 0xa9, 0xe5, 0x0e, 0x24, 0xdc, 0xca, 0x9e);

static const ble_uuid128_t char_uuid =
    BLE_UUID128_INIT(0x6e, 0x40, 0x00, 0x03, 0xb5, 0xa3, 0xf3, 0x93,
                     0xe0, 0xa9, 0xe5, 0x0e, 0x24, 0xdc, 0xca, 0x9e);

// GATT access callback
static int gatt_svr_chr_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                                   struct ble_gatt_access_ctxt *ctxt, void *arg) {
    if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        return os_mbuf_append(ctxt->om, &adc_value, sizeof(adc_value));
    }
    return 0;
}

// Notification timer task
void notify_task(void *param) {
    while (1) {
        if (subscribed && conn_handle != BLE_HS_CONN_HANDLE_NONE) {
            adc_value = adc1_get_raw(ADC1_CHANNEL_0); // GPIO36

            struct os_mbuf *om = ble_hs_mbuf_from_flat(&adc_value, sizeof(adc_value));
            int rc = ble_gattc_notify_custom(conn_handle, *((uint16_t *)param), om);

            if (rc != 0) {
                ESP_LOGE(TAG, "Notify failed: %d", rc);
            } else {
                ESP_LOGI(TAG, "Sent ADC: %d", adc_value);
            }
        } else {
            ESP_LOGI(TAG, "Waiting for subscription...");
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

// Called on connect/disconnect
static int ble_gap_event_cb(struct ble_gap_event *event, void *arg) {
    switch (event->type) {
        case BLE_GAP_EVENT_CONNECT:
            if (event->connect.status == 0) {
                ESP_LOGI(TAG, "Central connected");
                conn_handle = event->connect.conn_handle;
            } else {
                ESP_LOGI(TAG, "Connection failed; restarting advertising");
                ble_gap_adv_start(0, NULL, BLE_HS_FOREVER, NULL, ble_gap_event_cb, NULL);
            }
            return 0;

        case BLE_GAP_EVENT_DISCONNECT:
            ESP_LOGI(TAG, "Central disconnected");
            conn_handle = BLE_HS_CONN_HANDLE_NONE;
            subscribed = false;
            ble_gap_adv_start(0, NULL, BLE_HS_FOREVER, NULL, ble_gap_event_cb, NULL);
            return 0;

        case BLE_GAP_EVENT_SUBSCRIBE:
            subscribed = event->subscribe.cur_notify;
            ESP_LOGI(TAG, "Subscription changed: %s",
                     subscribed ? "subscribed" : "unsubscribed");
            return 0;

        default:
            return 0;
    }
}

// Define the GATT server
static void gatt_svr_init(void) {
    int rc;
    static uint16_t char_handle;

    ble_svc_gap_init();
    ble_svc_gatt_init();

    struct ble_gatt_svc_def gatt_svcs[] = {{
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &service_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {{
            .uuid = &char_uuid.u,
            .access_cb = gatt_svr_chr_access_cb,
            .val_handle = &char_handle,
            .flags = BLE_GATT_CHR_F_NOTIFY,
        }, {
            0, // end of characteristics
        }},
    }, {
        0, // end of services
    }};

    rc = ble_gatts_count_cfg(gatt_svcs);
    assert(rc == 0);

    rc = ble_gatts_add_svcs(gatt_svcs);
    assert(rc == 0);

    xTaskCreate(notify_task, "notify_task", 2048, &char_handle, 5, NULL);
}

// BLE host task
void ble_host_task(void *param) {
    nimble_port_run();
}

// Init BLE
void ble_init(void) {
    nvs_flash_init();
    esp_nimble_hci_and_controller_init();

    nimble_port_init();

    ble_hs_cfg.sync_cb = []() {
        ble_addr_t addr;
        ble_hs_id_infer_auto(0, &addr);
        char addr_str[18];
        ble_addr_to_str(&addr, addr_str);
        ESP_LOGI(TAG, "BLE address: %s", addr_str);

        ble_gap_adv_start(0, NULL, BLE_HS_FOREVER, NULL, ble_gap_event_cb, NULL);
    };

    gatt_svr_init();

    nimble_port_freertos_init(ble_host_task);
}

// Main app
extern "C" void app_main(void) {
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_0, ADC_ATTEN_DB_11);

    ble_init();
}
