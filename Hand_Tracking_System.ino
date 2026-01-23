#include <Servo.h>

// Define the number of angles (14 as per the Python code)
#define NUM_ANGLES 14

// Define servo pins (adjust based on your Arduino board; this assumes pins 2-15 for 14 servos)
// For Arduino Uno, you have limited PWM pins (3,5,6,9,10,11), so you may need to use a servo shield or Mega for more.
int servoPins[NUM_ANGLES] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, A0, A1}; // Example: pins 2-13 and A0-A1 (analog pins can be used as digital)

Servo servos[NUM_ANGLES];

String inputString = "";         // a String to hold incoming data
bool stringComplete = false;     // whether the string is complete

void setup() {
  Serial.begin(9600); // Match the baudrate from Python config
  inputString.reserve(200); // Reserve space for the input string

  // Attach servos
  for (int i = 0; i < NUM_ANGLES; i++) {
    servos[i].attach(servoPins[i]);
  }

  Serial.println("Arduino ready to receive angle data.");
}

void loop() {
  // Read serial data
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    if (inChar == '\n') {
      stringComplete = true;
    }
  }

  // If string is complete, parse and set servos
  if (stringComplete) {
    float angles[NUM_ANGLES];
    int index = 0;
    int start = 0;
    int commaIndex = inputString.indexOf(',');

    while (commaIndex != -1 && index < NUM_ANGLES) {
      String angleStr = inputString.substring(start, commaIndex);
      angles[index] = angleStr.toFloat();
      start = commaIndex + 1;
      commaIndex = inputString.indexOf(',', start);
      index++;
    }

    // Last angle (after last comma)
    if (index < NUM_ANGLES) {
      String angleStr = inputString.substring(start, inputString.length() - 1); // Remove newline
      angles[index] = angleStr.toFloat();
    }

    // Set servo positions (clamp to 0-180)
    for (int i = 0; i < NUM_ANGLES; i++) {
      int servoAngle = constrain((int)angles[i], 0, 180);
      servos[i].write(servoAngle);
    }

    // Optional: Print received angles for debugging
    Serial.print("Received angles: ");
    for (int i = 0; i < NUM_ANGLES; i++) {
      Serial.print(angles[i]);
      Serial.print(" ");
    }
    Serial.println();

    // Clear the string for next input
    inputString = "";
    stringComplete = false;
  }
}