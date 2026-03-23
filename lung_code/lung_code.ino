const int blowerPin = 10;

float BPM = 15.0;   // VERIFIED - 15-16 bpm
float inhaleRatio = 0.4;

unsigned long breathStartTime = 0;

void setup() {
  Serial.begin(115200);
  pinMode(blowerPin, OUTPUT);
}

void loop() {

  float T = 60.0 / BPM;
  float inhaleTime = T * inhaleRatio;

  unsigned long now = millis();
  float elapsed = (now - breathStartTime) / 1000.0;

  if (elapsed >= T) {
    breathStartTime = now;
    elapsed = 0;
  }

  float pwmValue = 0;

  if (elapsed < inhaleTime) {
    // Smooth inhale only
    float x = sin(PI * elapsed / inhaleTime);
    pwmValue = 220.0 * pow(x, 1.8);   // softened peak, capped at 220
  } 
  else {
    // Exhale = blower OFF
    pwmValue = 0;
  }

  pwmValue = constrain(pwmValue, 0, 255);
  analogWrite(blowerPin, (int)pwmValue);

  // SERIAL COMM. W/ PYTHON PROGRAM
  Serial.print(elapsed);
  Serial.print(",");
  Serial.print(elapsed / T);   // breath percent (0–1)
  Serial.print(",");
  Serial.println(pwmValue);
}
