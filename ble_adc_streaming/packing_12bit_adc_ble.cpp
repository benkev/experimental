//
// 
//
// MTU = Maximum Transmission Unit
// BLE 4.0 default MTU = 23 bytes total, with 20 bytes usable
// for actual data (3 bytes for protocol headers).
//

uint16_t adcSamples[10];  // 10 samples, each 12 bits
uint8_t packed[15];       // 15 bytes = 120 bits

// Fill adcSamples[] with analogRead() values (0–4095)

for (int i = 0; i < 10; ++i)
  adcSamples[i] = analogRead(36);

pCharacteristic->setValue((uint8_t*)samples, 20);
pCharacteristic->notify();

// Pack into packed[]:
int bitPos = 0;
for (int i = 0; i < 10; ++i) {
  uint16_t val = adcSamples[i] & 0x0FFF; // 12 bits
  int byteIdx = bitPos / 8;
  int bitOffset = bitPos % 8;

  packed[byteIdx]     |= val << bitOffset;
  packed[byteIdx + 1]  = (val >> (8 - bitOffset)) & 0xFF;
  if (bitOffset > 4)  // 12 bits may straddle 3 bytes
    packed[byteIdx + 2] = (val >> (16 - bitOffset)) & 0xFF;

  bitPos += 12;
}


// ===============================




