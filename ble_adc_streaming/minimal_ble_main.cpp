#include <Arduino.h>
#define  CONFIG_BT_NIMBLE_NVS_PERSIST 0
#include <NimBLEDevice.h>

#define SERVICE_UUID        "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define CHARACTERISTIC_UUID "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

NimBLECharacteristic *pCharacteristic;
bool deviceConnected = false;

class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer *pServer) {
    Serial.println("Central connected");
    deviceConnected = true;
  }
  void onDisconnect(NimBLEServer *pServer) {
    Serial.println("Central disconnected");
    deviceConnected = false;
  }
};

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);

  NimBLEDevice::init("ESP32-BLE-ADC-Minimal");
  NimBLEServer *pServer = NimBLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  NimBLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    NIMBLE_PROPERTY::NOTIFY
  );
  
  NimBLEAdvertising *pAdvertising = NimBLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->start();
}

void loop() {
  if (deviceConnected) {
    uint16_t adc = analogRead(36);
    pCharacteristic->setValue((uint8_t*)&adc, sizeof(adc));
    pCharacteristic->notify();
    Serial.print("Sent ADC: ");
    Serial.println(adc);
    delay(10);
  } else {
    Serial.println("Waiting for connection...");
    delay(100);
  }
}
