//
// ADC1 (pin 36) 12-bit data streaming via BLE
//
// With recovery after disconnection, and smooth reconnection
//
//

#include <Arduino.h>
#define CONFIG_BT_NIMBLE_NVS_PERSIST 0
#include <NimBLEDevice.h>

#define SERVICE_UUID        "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define CHARACTERISTIC_UUID "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

NimBLECharacteristic *pCharacteristic;
bool subscribed = false;  // Set true only after client subscribes

// GAP-level callbacks (connection/disconnection)
class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* pServer) {
    Serial.println("Central connected (GAP level)");
  }

  void onDisconnect(NimBLEServer* pServer) {
    Serial.println("Central disconnected");
    subscribed = false;

    // Restart advertising so new clients can connect
    NimBLEDevice::getAdvertising()->start();
    Serial.println("Advertising restarted");
  }
};

// GATT-level callbacks for subscription
class CharacteristicCallbacks : public NimBLECharacteristicCallbacks {
  void onSubscribe(NimBLECharacteristic* pCharacteristic,
                   NimBLEConnInfo& connInfo, uint16_t subValue) {
    Serial.println("Client subscribed to notifications");
    subscribed = true;
  }

  void onUnsubscribe(NimBLECharacteristic* pCharacteristic,
                     NimBLEConnInfo& connInfo) {
    Serial.println("Client unsubscribed");
    subscribed = false;
  }
};

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);

  NimBLEDevice::init("ESP32-BLE-ADC-Minimal");

  // Optional: disable security (avoid pairing issues on Linux/BlueZ)
  NimBLEDevice::setSecurityAuth(false, false, false);

  NimBLEServer *pServer = NimBLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  NimBLEService *pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    NIMBLE_PROPERTY::NOTIFY
  );
  pCharacteristic->setCallbacks(new CharacteristicCallbacks());

  pService->start();

  NimBLEAdvertising *pAdvertising = NimBLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->start();

  Serial.println("Waiting for client to connect and subscribe...");
}

void loop() {
  static uint32_t lastCheck = 0;
  if (millis() - lastCheck > 1000) {
    lastCheck = millis();
    
    // Detect whether the ESP32 is still connected to any client after
    // a Ctrl+C on the PC
    if (NimBLEDevice::getServer()->getConnectedCount() == 0) {
      if (subscribed) {
        Serial.println("No clients connected but still marked subscribed." \
                       " Resetting state.");
        subscribed = false;
        NimBLEDevice::startAdvertising();
        Serial.println("Advertising restarted");
      }
    }
  }

  if (subscribed) {
    uint16_t adc = analogRead(36);
    pCharacteristic->setValue((uint8_t*)&adc, sizeof(adc));
    pCharacteristic->notify();

    Serial.print("Sent ADC: ");
    Serial.println(adc);
    delay(10);
  } else {
    Serial.println("Waiting for subscription...");
    delay(500);
  }
}
