//
// Code for Wemos ESP32 with OLED screen that plots ADC0 (GPIO36) voltage vs time
// The board: Wemos Lolin32
//

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

#define ADC_PIN 36  // GPIO36 = ADC0 = SVP

int graph[SCREEN_WIDTH]; // Circular buffer

void setup() {
  analogReadResolution(12); // Default is 12 bits on ESP32

  Serial.begin(115200);
  
  // Important: initialize I2C with correct OLED pins
  // common on Wemos ESP32 OLED boards:
  Wire.begin(5, 4);  // SDA = GPIO5, SCL = GPIO4

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("OLED init failed"));
    for (;;);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("ADC Plotting...");
  display.display();
  delay(1000);
}

void loop() {
  int value = analogRead(ADC_PIN); // 0-4095
  float voltage = (value / 4095.0) * 3.3; // Convert to volts

  // Shift graph left
  for (int i = 0; i < SCREEN_WIDTH - 1; i++) {
    graph[i] = graph[i + 1];
  }

  // Scale voltage to screen height
  graph[SCREEN_WIDTH - 1] = SCREEN_HEIGHT - 1 - int((voltage / 3.3) * SCREEN_HEIGHT);

  // Draw graph
  display.clearDisplay();
  for (int x = 0; x < SCREEN_WIDTH - 1; x++) {
    display.drawLine(x, graph[x], x + 1, graph[x + 1], SSD1306_WHITE);
  }

  // Show voltage
  display.setCursor(0, 0);
  display.print("V = ");
  display.print(voltage, 2);
  display.println(" V");
  display.display();

  delay(100); // 10 Hz sampling
}
