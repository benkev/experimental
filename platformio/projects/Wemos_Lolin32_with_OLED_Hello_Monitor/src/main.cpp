#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// OLED display size
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

// SDA = GPIO 5, SCL = GPIO 4 (specific to your board)
// OLED_RESET is not connected
#define OLED_RESET     -1

// Create display object
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

void setup() {
  // Start serial monitor
  Serial.begin(115200);
  delay(1000); // Short delay to allow Serial Monitor to catch up

  // Initialize I2C with correct pins
  Wire.begin(5, 4);

  // Initialize the OLED display
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED initialization failed");
    while (true); // Stay here if OLED init fails
  }

  // Clear display and set text properties
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.print("Hello!");
  display.display();

  Serial.println("OLED shows: Hello!");
}

void loop() {
  // Blink the OLED and print to serial
  display.clearDisplay();
  display.display();
  Serial.println("OLED cleared");
  delay(500);

  display.setCursor(0, 0);
  display.print("Hello!");
  display.display();
  Serial.println("OLED shows: Hello!");
  delay(500);
}
