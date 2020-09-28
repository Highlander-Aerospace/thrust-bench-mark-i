#include <Arduino.h>
#include <LiquidCrystal_I2C.h>
#include <HX711.h>

//LiquidCrystal_I2C lcd(0x27, 16, 4);
HX711 loadcell;

const int LOADCELL_DOUT_PIN = 2;
const int LOADCELL_SCK_PIN = 3;
const float CALIBRATION_SCALE = -2467.8353715336484129;
bool recording = false;
String msg;

void setup() {
    Serial.begin(115200);
    Serial.println("HIAERO MARK I THRUST BENCH SOFTWARE BOOTING..");
    loadcell.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
    loadcell.set_scale(CALIBRATION_SCALE);
}

void loop() {
    if (!recording) {
        msg = Serial.readStringUntil('\n');
        if (msg == "I/O TEST FROM MASTER") {
            Serial.println("I/O RESPONSE FROM HARDWARE");
        } else if (msg == "R") {
            recording = true;
            Serial.println("R");
        } else if (msg == "T") {
            loadcell.tare();
            Serial.println("T");
        }
    } else {
        if (Serial.available() > 0) {
            msg = Serial.readStringUntil('\n');
            if (msg == "S") {
                recording = false;
                Serial.println("S");
            }
        } else {
            Serial.println(loadcell.get_units(1));
        }
    }
}
