#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>

// BLE Service and Characteristic UUIDs
#define SERVICE_UUID        "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define CHARACTERISTIC_UUID "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

BLECharacteristic *pCharacteristic;
bool deviceConnected = false;

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) {
    deviceConnected = true;
    Serial.println("Central connected");
  }

  void onDisconnect(BLEServer *pServer) {
    deviceConnected = false;
    Serial.println("Central disconnected");
  }
};

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);  // 0–4095

  BLEDevice::init("ESP32-BLE-ADC");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_NOTIFY
                    );

  // Add Client Characteristic Configuration Descriptor (CCCD)
  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();

  pServer->getAdvertising()->start();
  delay(1000); 
  Serial.println("BLE ADC service is ready!");
}

void loop() {
  if (deviceConnected) {
    int adcValue = analogRead(36);  // ADC1_CH0 (GPIO36)
    uint16_t value = adcValue;
    pCharacteristic->setValue((uint8_t*)&value, sizeof(value));
    pCharacteristic->notify();

    delay(10);  // 100 Hz sampling
  } else {
    delay(100);
  }
}
