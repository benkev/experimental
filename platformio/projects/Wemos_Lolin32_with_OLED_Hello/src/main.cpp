#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET     -1  // Reset pin # (or -1 if sharing Arduino reset pin)
#define OLED_SDA       5
#define OLED_SCL       4

// Set up I2C using custom pins
TwoWire myWire = TwoWire(0);
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &myWire, OLED_RESET);

void setup() {
  Serial.begin(115200);

  myWire.begin(OLED_SDA, OLED_SCL);

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // Address 0x3C for 128x64
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // Don't proceed, loop forever
  }

  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(F("Hello!"));
  display.display();
}

void loop() {
  delay(1000);
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println(F("Hello!"));
  display.display();
  delay(1000);
  display.clearDisplay();
  display.display();
}
